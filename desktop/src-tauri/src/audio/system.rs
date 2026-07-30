//! The machine's own output — the "Meeting" side.
//!
//! Port of `audio_capture.SystemCapture` / `ScreenCaptureKitCapture` /
//! `LoopbackDeviceCapture` / `open_system_capture`.
//!
//! Everyone other than you arrives as *playback*: Zoom, Teams and Meet decode
//! the far end and send it to the speakers, where an ordinary input device can't
//! reach it. Two backends can, and [`open_system_capture`] picks between them in
//! the same order the Python app did:
//!
//! 1. **ScreenCaptureKit** (macOS 13+) — taps the system mix directly. Needs the
//!    Screen Recording permission.
//! 2. **A loopback input device** — BlackHole, Loopback, VB-Audio and friends.
//!    The fallback for an older macOS, Linux, Windows, or a denied permission.
//!
//! The contract that matters is **report, don't raise**: if the meeting side
//! can't be recorded, the microphone still can, so a failure here becomes a
//! warning on the recording rather than an error that loses the take. A meeting
//! is not repeatable, and half a recording beats none.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, StreamConfig};

use super::mic::BlockSink;
use super::resampler::Resampler;
use super::to_mono;
use crate::SAMPLE_RATE;

/// Substrings of device names that are loopback devices rather than real inputs.
/// Matched case-insensitively, same list as the Python app's `KNOWN`.
const LOOPBACK_NAMES: &[&str] = &[
    "blackhole",
    "soundflower",
    "loopback",
    "vb-audio",
    "cable output",
    "stereo mix",
    "monitor of",
];

/// Which backend took the meeting side, for the UI to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemBackend {
    ScreenCaptureKit,
    LoopbackDevice,
}

impl SystemBackend {
    pub fn description(self) -> &'static str {
        match self {
            SystemBackend::ScreenCaptureKit => "macOS system audio (ScreenCaptureKit)",
            SystemBackend::LoopbackDevice => "system audio via a loopback input device",
        }
    }
}

/// A running system-audio capture. Dropping it stops the capture.
pub struct SystemCapture {
    backend: SystemBackend,
    /// cpal stream, for the loopback backend.
    _stream: Option<cpal::Stream>,
    #[cfg(target_os = "macos")]
    sck: Option<sck::ScreenCaptureKitCapture>,
    error: Arc<Mutex<Option<String>>>,
}

impl SystemCapture {
    pub fn backend(&self) -> SystemBackend {
        self.backend
    }

    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }

    pub fn stop(&mut self) {
        self._stream = None;
        #[cfg(target_os = "macos")]
        if let Some(capture) = self.sck.as_mut() {
            capture.stop();
        }
    }
}

/// Which backend to try. `Auto` is what the UI uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Prefer {
    #[default]
    Auto,
    ScreenCaptureKit,
    Loopback,
}

/// Pick a backend for the system mix.
///
/// Returns `(capture, problem)`. A `None` capture with a problem message means
/// the meeting side can't be recorded on this machine right now — the microphone
/// still can, which is why this reports rather than raises.
pub fn open_system_capture(
    prefer: Prefer,
    clock: Instant,
    sink: Box<BlockSink>,
) -> (Option<SystemCapture>, Option<String>) {
    let want_sck = matches!(prefer, Prefer::Auto | Prefer::ScreenCaptureKit);
    let want_loopback = matches!(prefer, Prefer::Auto | Prefer::Loopback);

    let mut sink = Some(sink);
    let mut problem;

    #[cfg(target_os = "macos")]
    {
        if want_sck {
            match sck::ScreenCaptureKitCapture::start(clock, sink.take().unwrap()) {
                Ok((capture, error)) => {
                    return (
                        Some(SystemCapture {
                            backend: SystemBackend::ScreenCaptureKit,
                            _stream: None,
                            sck: Some(capture),
                            error,
                        }),
                        None,
                    );
                }
                Err((message, returned)) => {
                    sink = Some(returned);
                    problem = Some(message);
                }
            }
        } else {
            problem = Some("System audio capture was not requested.".to_string());
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = want_sck;
        problem = Some("System audio capture is not available on this platform.".to_string());
    }

    if want_loopback {
        match start_loopback(clock, sink.take().unwrap()) {
            Ok(capture) => return (Some(capture), None),
            Err(error) => {
                // The loopback problem is the more actionable one when the user
                // has no permission *and* no loopback device, so it wins.
                problem = Some(match problem {
                    Some(first) => format!("{first} {error}"),
                    None => error.to_string(),
                });
            }
        }
    }

    (None, problem)
}

/// Find a loopback input device, if one is installed.
pub fn find_loopback_device() -> Option<cpal::Device> {
    let host = cpal::default_host();
    let devices = host.input_devices().ok()?;
    devices.into_iter().find(|device| {
        device
            .description()
            .ok()
            .map(|description| is_loopback_name(description.name()))
            .unwrap_or(false)
    })
}

/// Whether a device name looks like a loopback driver rather than a real input.
pub fn is_loopback_name(name: &str) -> bool {
    let lowered = name.to_lowercase();
    LOOPBACK_NAMES.iter().any(|needle| lowered.contains(needle))
}

fn start_loopback(clock: Instant, mut sink: Box<BlockSink>) -> Result<SystemCapture> {
    let device = find_loopback_device().ok_or_else(|| {
        anyhow!(
            "No loopback audio device was found. Install BlackHole (or similar) \
             and route playback through it to record the meeting side."
        )
    })?;

    let config = device
        .default_input_config()
        .context("the loopback device did not report a usable configuration")?;
    let channels = config.channels() as usize;
    let rate = config.sample_rate();
    let mut resampler = (rate != SAMPLE_RATE).then(|| Resampler::new(rate));

    let error = Arc::new(Mutex::new(None));
    let error_for_callback = Arc::clone(&error);

    let mut deliver = move |samples: &[f32]| {
        let frames = samples.len() / channels.max(1);
        let started_at = clock.elapsed().as_secs_f64() - frames as f64 / rate as f64;
        let mono = to_mono(samples, channels);
        let block = match resampler.as_mut() {
            Some(resampler) => resampler.process(&mono),
            None => mono,
        };
        if !block.is_empty() {
            sink(&block, started_at);
        }
    };

    let on_error = move |err: cpal::Error| {
        *error_for_callback.lock().unwrap() = Some(err.to_string());
    };

    let stream_config = StreamConfig {
        channels: config.channels(),
        sample_rate: rate,
        buffer_size: BufferSize::Default,
    };

    let stream = match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            stream_config,
            move |data: &[f32], _| deliver(data),
            on_error,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            stream_config,
            move |data: &[i16], _| {
                let converted: Vec<f32> =
                    data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                deliver(&converted)
            },
            on_error,
            None,
        ),
        other => return Err(anyhow!("unsupported loopback sample format {other:?}")),
    }
    .context("could not open the loopback device")?;

    stream.play().context("could not start the loopback device")?;

    Ok(SystemCapture {
        backend: SystemBackend::LoopbackDevice,
        _stream: Some(stream),
        #[cfg(target_os = "macos")]
        sck: None,
        error,
    })
}

#[cfg(target_os = "macos")]
mod sck {
    //! ScreenCaptureKit backend.
    //!
    //! Configured to hand us exactly what the pipeline wants — 16 kHz, mono —
    //! so CoreAudio does the rate conversion rather than us, and to exclude
    //! this process's own audio so the app can never record itself.

    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use screencapturekit::cm::CMSampleBuffer;
    use screencapturekit::stream::{
        configuration::SCStreamConfiguration, content_filter::SCContentFilter,
        output_trait::SCStreamOutputTrait, output_type::SCStreamOutputType, SCStream,
    };
    use screencapturekit::shareable_content::SCShareableContent;

    use super::super::mic::BlockSink;
    use crate::SAMPLE_RATE;

    /// Bridges SCK's delegate callback to our block sink.
    struct AudioOutput {
        clock: Instant,
        sink: Mutex<Box<BlockSink>>,
        error: Arc<Mutex<Option<String>>>,
    }

    impl SCStreamOutputTrait for AudioOutput {
        fn did_output_sample_buffer(
            &self,
            sample_buffer: CMSampleBuffer,
            of_type: SCStreamOutputType,
        ) {
            if of_type != SCStreamOutputType::Audio {
                return;
            }
            let Some(samples) = mono_samples(&sample_buffer) else {
                return;
            };
            if samples.is_empty() {
                return;
            }
            // Same reasoning as the microphone: the callback runs after the
            // block was captured, so its first sample belongs one block back.
            let started_at =
                self.clock.elapsed().as_secs_f64() - samples.len() as f64 / SAMPLE_RATE as f64;

            match self.sink.lock() {
                Ok(mut sink) => sink(&samples, started_at),
                Err(_) => {
                    *self.error.lock().unwrap() =
                        Some("the system audio sink panicked".to_string());
                }
            }
        }
    }

    /// Mono float32 samples out of one ScreenCaptureKit audio sample buffer.
    ///
    /// SCK delivers non-interleaved float32 — channel after channel, not sample
    /// after sample — so each buffer in the list is one channel and the mean
    /// across them is the mix we want.
    fn mono_samples(sample_buffer: &CMSampleBuffer) -> Option<Vec<f32>> {
        let list = sample_buffer.audio_buffer_list()?;
        let channels = list.num_buffers();
        if channels == 0 {
            return None;
        }

        let mut planes: Vec<Vec<f32>> = Vec::with_capacity(channels);
        for index in 0..channels {
            let buffer = list.buffer(index)?;
            planes.push(f32_from_bytes(buffer.data()));
        }

        let frames = planes.iter().map(Vec::len).min().unwrap_or(0);
        if frames == 0 {
            return None;
        }
        if channels == 1 {
            let mut plane = planes.pop()?;
            plane.truncate(frames);
            return Some(plane);
        }

        Some(
            (0..frames)
                .map(|frame| {
                    planes.iter().map(|plane| plane[frame]).sum::<f32>() / channels as f32
                })
                .collect(),
        )
    }

    fn f32_from_bytes(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
            .collect()
    }

    pub struct ScreenCaptureKitCapture {
        stream: SCStream,
    }

    impl ScreenCaptureKitCapture {
        /// Start capturing the system mix.
        ///
        /// On failure the sink is handed back so the caller can try the loopback
        /// backend with it — the closure is not cloneable, and losing it would
        /// mean losing the fallback.
        #[allow(clippy::type_complexity)]
        pub fn start(
            clock: Instant,
            sink: Box<BlockSink>,
        ) -> Result<(Self, Arc<Mutex<Option<String>>>), (String, Box<BlockSink>)> {
            // Asking for shareable content is also the permission check: it fails
            // when Screen Recording has not been granted.
            let content = match SCShareableContent::get() {
                Ok(content) => content,
                Err(error) => {
                    return Err((
                        format!(
                            "Screen Recording permission is not granted, so the meeting \
                             side can't be recorded ({error}). Grant it in System Settings \
                             > Privacy & Security > Screen & System Audio Recording, then \
                             restart this app."
                        ),
                        sink,
                    ));
                }
            };

            let Some(display) = content.displays().into_iter().next() else {
                return Err(("no display was available to capture audio from".into(), sink));
            };

            let filter = SCContentFilter::create().with_display(&display).build();

            let mut configuration = SCStreamConfiguration::new();
            configuration
                .set_captures_audio(true)
                // Let CoreAudio resample: its converter is better than ours and
                // costs nothing, exactly as on the microphone path.
                .set_sample_rate(SAMPLE_RATE as i32)
                .set_channel_count(1)
                // Never record our own output. Without this the app's own sounds
                // would land in the "Meeting" transcript.
                .set_excludes_current_process_audio(true);

            let error = Arc::new(Mutex::new(None));
            let mut stream = SCStream::new(&filter, &configuration);
            stream.add_output_handler(
                AudioOutput {
                    clock,
                    sink: Mutex::new(sink),
                    error: Arc::clone(&error),
                },
                SCStreamOutputType::Audio,
            );

            if let Err(problem) = stream.start_capture() {
                // The sink is inside the handler now and cannot be recovered, so
                // report without it; the caller falls back with a fresh sink.
                return Err((
                    format!("could not start system audio capture: {problem}"),
                    Box::new(|_: &[f32], _: f64| {}),
                ));
            }

            Ok((Self { stream }, error))
        }

        pub fn stop(&mut self) {
            // Best effort: a stream that refuses to stop is about to be dropped.
            let _ = self.stream.stop_capture();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_loopback_drivers_are_recognised() {
        assert!(is_loopback_name("BlackHole 2ch"));
        assert!(is_loopback_name("blackhole 16ch"));
        assert!(is_loopback_name("Loopback Audio"));
        assert!(is_loopback_name("VB-Audio Virtual Cable"));
        assert!(is_loopback_name("CABLE Output (VB-Audio Point)"));
        assert!(is_loopback_name("Stereo Mix"));
        assert!(is_loopback_name("Monitor of Built-in Audio"));
        assert!(is_loopback_name("Soundflower (2ch)"));
    }

    #[test]
    fn real_inputs_are_not_mistaken_for_loopback() {
        // Recording the microphone into the "Meeting" track would double every
        // word the local speaker says.
        assert!(!is_loopback_name("MacBook Pro Microphone"));
        assert!(!is_loopback_name("AirPods Pro"));
        assert!(!is_loopback_name("External Headset"));
        assert!(!is_loopback_name("Studio Display Microphone"));
    }

    #[test]
    fn backend_descriptions_say_where_the_audio_came_from() {
        assert!(SystemBackend::ScreenCaptureKit
            .description()
            .contains("ScreenCaptureKit"));
        assert!(SystemBackend::LoopbackDevice
            .description()
            .contains("loopback"));
    }

    #[test]
    fn a_missing_backend_reports_rather_than_panics() {
        // The report-don't-raise contract. Whatever this machine has, asking for
        // a loopback device that isn't there must yield a message, never a panic
        // and never a lost recording.
        let (capture, problem) = open_system_capture(
            Prefer::Loopback,
            Instant::now(),
            Box::new(|_: &[f32], _: f64| {}),
        );
        match (capture, problem) {
            (Some(capture), None) => {
                assert_eq!(capture.backend(), SystemBackend::LoopbackDevice);
            }
            (None, Some(message)) => {
                assert!(!message.is_empty(), "a failure must explain itself");
            }
            other => panic!("expected exactly one of capture/problem, got {:?}", other.1),
        }
    }
}
