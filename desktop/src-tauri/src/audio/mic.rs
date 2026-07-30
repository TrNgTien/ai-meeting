//! Microphone -> mono 16 kHz.
//!
//! Port of `audio_capture.MicCapture`, on cpal instead of PortAudio.
//!
//! Two behaviours from the Python version are load-bearing and carried over:
//!
//! * **Ask the OS for 16 kHz directly.** CoreAudio's own rate conversion is
//!   better than ours and costs nothing; we only fall back to
//!   [`Resampler`](super::resampler::Resampler) on devices that refuse the rate.
//! * **Timestamp the block at its *start*.** The callback runs *after* the block
//!   was captured, so its first sample belongs one block-length back on the
//!   timeline. Getting this wrong pushes one track permanently later than the
//!   other, which is precisely the drift the WAV writer exists to prevent.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, StreamConfig};

use super::devices::resolve_input_device;
use super::resampler::Resampler;
use super::to_mono;
use crate::SAMPLE_RATE;

/// Receives mono 16 kHz blocks plus the wall-clock offset of their first sample.
pub type BlockSink = dyn FnMut(&[f32], f64) + Send + 'static;

pub struct MicCapture {
    /// Dropping the stream stops it; held only to own it.
    _stream: cpal::Stream,
    /// Last error the audio callback saw, for the UI's warnings list.
    error: Arc<Mutex<Option<String>>>,
}

impl MicCapture {
    /// Open the microphone and start delivering blocks to `sink`.
    ///
    /// `clock` is the shared time origin: every capture backend measures its
    /// offsets against the same `Instant`, which is what lets the two tracks
    /// share one timeline.
    pub fn start(
        device_id: Option<&str>,
        clock: Instant,
        mut sink: Box<BlockSink>,
    ) -> Result<Self> {
        let device =
            resolve_input_device(device_id).ok_or_else(|| anyhow!("no microphone is available"))?;

        let default_config = device
            .default_input_config()
            .context("the microphone did not report a usable configuration")?;
        let channels = default_config.channels() as usize;
        let sample_format = default_config.sample_format();

        // Prefer 16 kHz straight from the device; fall back to its own rate.
        let native_rate = default_config.sample_rate();
        let (rate, mut resampler) = if supports_16k(&device) {
            (SAMPLE_RATE, None)
        } else {
            (native_rate, Some(Resampler::new(native_rate)))
        };

        let config = StreamConfig {
            channels: default_config.channels(),
            sample_rate: rate,
            buffer_size: BufferSize::Default,
        };

        let error = Arc::new(Mutex::new(None));
        let error_for_callback = Arc::clone(&error);

        // One closure, shared by every sample format: convert to mono f32,
        // resample if needed, stamp the block's start, hand it on.
        let mut deliver = move |samples: &[f32]| {
            let frames = samples.len() / channels.max(1);
            // The callback runs after the block was captured.
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
            // Overflows mean the machine couldn't keep up; the writer's resync
            // turns the lost audio into silence of the right length rather than
            // a shift in everything that follows.
            *error_for_callback.lock().unwrap() = Some(err.to_string());
        };

        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                config,
                move |data: &[f32], _| deliver(data),
                on_error,
                None,
            ),
            SampleFormat::I16 => device.build_input_stream(
                config,
                move |data: &[i16], _| {
                    let converted: Vec<f32> =
                        data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                    deliver(&converted)
                },
                on_error,
                None,
            ),
            SampleFormat::U16 => device.build_input_stream(
                config,
                move |data: &[u16], _| {
                    let converted: Vec<f32> = data
                        .iter()
                        .map(|s| (*s as f32 - 32768.0) / 32768.0)
                        .collect();
                    deliver(&converted)
                },
                on_error,
                None,
            ),
            other => return Err(anyhow!("unsupported microphone sample format {other:?}")),
        }
        .context("could not open the microphone")?;

        stream.play().context("could not start the microphone")?;

        Ok(Self {
            _stream: stream,
            error,
        })
    }

    /// The last callback error, if any — surfaced in `Recording::warnings`.
    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }
}

/// Whether the device will give us 16 kHz without our own conversion.
fn supports_16k(device: &cpal::Device) -> bool {
    device
        .supported_input_configs()
        .map(|mut ranges| ranges.any(|range| range.contains_rate(SAMPLE_RATE)))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_block_is_stamped_at_its_start_not_its_arrival() {
        // The arithmetic that keeps the two tracks together. A 1024-frame block
        // at 48 kHz was captured over ~21.3 ms, so its first sample belongs that
        // far back from when the callback ran.
        let elapsed = 5.0f64;
        let frames = 1024.0f64;
        let rate = 48_000.0f64;
        let started_at = elapsed - frames / rate;
        assert!((started_at - 4.978_666).abs() < 1e-5, "got {started_at}");
        assert!(
            started_at < elapsed,
            "a block can never start after the callback that delivered it"
        );
    }

    #[test]
    fn u16_samples_map_to_the_full_signed_range() {
        // Mid-scale is silence; the extremes are +/- full scale.
        let convert = |s: u16| (s as f32 - 32768.0) / 32768.0;
        assert_eq!(convert(32768), 0.0);
        assert_eq!(convert(0), -1.0);
        assert!((convert(65535) - 0.999_97).abs() < 1e-4);
    }

    #[test]
    fn i16_samples_map_to_unit_range() {
        let convert = |s: i16| s as f32 / i16::MAX as f32;
        assert_eq!(convert(0), 0.0);
        assert_eq!(convert(i16::MAX), 1.0);
        assert!((convert(i16::MIN) + 1.0).abs() < 1e-4);
    }
}
