//! Rate conversion that keeps its filter warm across callbacks.
//!
//! Port of `audio_capture._Resampler`, which leaned on `scipy.signal.resample_poly`.
//! The behaviour that matters is in that class's docstring:
//!
//! > Resampling each block in isolation leaves a discontinuity at every boundary
//! > — a faint buzz at the block rate, which is exactly the kind of artefact
//! > Whisper turns into invented words.
//!
//! The Python version carried 256 samples of input history and threw away the
//! output they produced. This is a proper streaming polyphase resampler instead:
//! it keeps exactly the filter state it needs and emits a continuous signal, so
//! there is no boundary to smooth over and no lead-in to discard.
//!
//! Written out rather than taken from a crate because `rubato` 4.0 landed a
//! ground-up API rewrite built on `audioadapter` buffers, and a hand-rolled
//! upfirdn is both smaller and easier to test than that integration would be.
//! The tests check the properties that actually matter — passband, stopband,
//! gain, and continuity across block boundaries.

use crate::SAMPLE_RATE;

/// Half-length of the anti-aliasing filter, in output-phase units. 16 gives a
/// ~97-tap filter for the common 48 kHz -> 16 kHz case: sharp enough to keep
/// aliases well down, cheap enough to run in an audio callback.
const HALF_LEN: usize = 16;

/// Rational resampler from an arbitrary input rate to [`SAMPLE_RATE`].
pub struct Resampler {
    up: usize,
    down: usize,
    /// Anti-aliasing FIR, designed at the upsampled rate, scaled so DC gain is 1.
    taps: Vec<f32>,
    /// Input samples still needed by future outputs. `buf[0]` is input index
    /// `input_base`.
    buf: Vec<f32>,
    input_base: u64,
    next_out: u64,
}

impl Resampler {
    pub fn new(src_rate: u32) -> Self {
        let divisor = gcd(src_rate as usize, SAMPLE_RATE as usize);
        let up = SAMPLE_RATE as usize / divisor;
        let down = src_rate as usize / divisor;

        let taps = if up == 1 && down == 1 {
            Vec::new()
        } else {
            design_lowpass(up, down)
        };

        Self {
            up,
            down,
            taps,
            buf: Vec::new(),
            input_base: 0,
            next_out: 0,
        }
    }

    /// True when the input is already at [`SAMPLE_RATE`] and blocks pass through.
    pub fn is_passthrough(&self) -> bool {
        self.up == 1 && self.down == 1
    }

    /// Resample one block, continuing seamlessly from the previous call.
    pub fn process(&mut self, block: &[f32]) -> Vec<f32> {
        if self.is_passthrough() {
            return block.to_vec();
        }
        if block.is_empty() {
            return Vec::new();
        }

        self.buf.extend_from_slice(block);
        let taps_len = self.taps.len() as u64;
        let up = self.up as u64;
        let down = self.down as u64;
        let available_end = self.input_base + self.buf.len() as u64;

        let mut out = Vec::new();
        loop {
            // Output n draws on input indices k where 0 <= n*down - k*up < taps_len,
            // so the newest input it needs is floor(n*down / up).
            let newest_needed = self.next_out * down / up;
            if newest_needed >= available_end {
                break;
            }

            let offset = self.next_out * down;
            // Oldest input this output touches.
            let oldest_needed = (offset + 1).saturating_sub(taps_len).div_ceil(up);

            let mut sum = 0.0f32;
            for k in oldest_needed..=newest_needed {
                if k < self.input_base {
                    continue;
                }
                let tap = offset - k * up;
                if tap >= taps_len {
                    continue;
                }
                sum += self.buf[(k - self.input_base) as usize] * self.taps[tap as usize];
            }
            out.push(sum);
            self.next_out += 1;
        }

        // Drop input the next output can no longer reach.
        let next_offset = self.next_out * down;
        let keep_from = (next_offset + 1).saturating_sub(taps_len).div_ceil(up);
        if keep_from > self.input_base {
            let drop = (keep_from - self.input_base) as usize;
            let drop = drop.min(self.buf.len());
            self.buf.drain(..drop);
            self.input_base += drop as u64;
        }

        out
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a.max(1)
    } else {
        gcd(b, a % b)
    }
}

/// Windowed-sinc lowpass for polyphase resampling by `up`/`down`.
///
/// Cutoff sits at half the *lower* of the two rates, which is what prevents
/// aliasing when decimating and imaging when interpolating. Normalised so the
/// filter's DC gain through the up/down pair is exactly 1 — otherwise every
/// recording would come out quieter or louder than it went in.
fn design_lowpass(up: usize, down: usize) -> Vec<f32> {
    let max_rate = up.max(down);
    let half = HALF_LEN * max_rate;
    let len = 2 * half + 1;

    let mut taps = vec![0.0f32; len];
    for (index, tap) in taps.iter_mut().enumerate() {
        let position = index as f64 - half as f64;
        let sinc = if position == 0.0 {
            1.0
        } else {
            let x = std::f64::consts::PI * position / max_rate as f64;
            x.sin() / x
        };
        // Blackman window: deeper stopband than Hamming, which matters because
        // an alias folded into the speech band is indistinguishable from speech.
        let ratio = index as f64 / (len - 1) as f64;
        let window = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * ratio).cos()
            + 0.08 * (4.0 * std::f64::consts::PI * ratio).cos();
        *tap = (sinc * window) as f32;
    }

    // Each output sample sees every `up`th tap, so the per-phase sum is what
    // needs to come to 1.
    let sum: f32 = taps.iter().step_by(1).sum();
    if sum != 0.0 {
        let scale = up as f32 / sum;
        for tap in &mut taps {
            *tap *= scale;
        }
    }
    taps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    fn sine(rate: u32, freq: f64, seconds: f64) -> Vec<f32> {
        let count = (rate as f64 * seconds) as usize;
        (0..count)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64).sin() as f32
            })
            .collect()
    }

    /// Dominant frequency by brute-force correlation — enough to tell 440 Hz from
    /// an alias without pulling in an FFT crate.
    fn dominant_freq(samples: &[f32], rate: u32, candidates: &[f64]) -> f64 {
        let mut best = (0.0f64, f64::NEG_INFINITY);
        for &freq in candidates {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (i, sample) in samples.iter().enumerate() {
                let phase = 2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64;
                re += *sample as f64 * phase.cos();
                im += *sample as f64 * phase.sin();
            }
            let power = re * re + im * im;
            if power > best.1 {
                best = (freq, power);
            }
        }
        best.0
    }

    #[test]
    fn a_matching_rate_passes_straight_through() {
        let mut resampler = Resampler::new(SAMPLE_RATE);
        assert!(resampler.is_passthrough());
        let block = vec![0.1, -0.2, 0.3];
        assert_eq!(resampler.process(&block), block);
    }

    #[test]
    fn common_device_rates_reduce_to_small_ratios() {
        let r48 = Resampler::new(48_000);
        assert_eq!((r48.up, r48.down), (1, 3));
        let r44 = Resampler::new(44_100);
        assert_eq!((r44.up, r44.down), (160, 441));
        let r32 = Resampler::new(32_000);
        assert_eq!((r32.up, r32.down), (1, 2));
    }

    #[test]
    fn output_length_tracks_the_rate_ratio() {
        let mut resampler = Resampler::new(48_000);
        // One second in at 48 kHz should be about one second out at 16 kHz.
        let out = resampler.process(&sine(48_000, 440.0, 1.0));
        let expected = 16_000i64;
        assert!(
            (out.len() as i64 - expected).abs() < 40,
            "got {} samples, expected ~{expected}",
            out.len()
        );
    }

    #[test]
    fn a_speech_band_tone_survives_at_the_same_frequency() {
        let mut resampler = Resampler::new(48_000);
        let out = resampler.process(&sine(48_000, 440.0, 0.5));
        // Skip the filter's lead-in before measuring.
        let steady = &out[400..];
        let candidates: Vec<f64> = (100..2000).step_by(10).map(|f| f as f64).collect();
        let found = dominant_freq(steady, SAMPLE_RATE, &candidates);
        assert!((found - 440.0).abs() <= 10.0, "found {found} Hz");
    }

    #[test]
    fn amplitude_is_preserved() {
        let mut resampler = Resampler::new(48_000);
        let input = sine(48_000, 440.0, 0.5);
        let out = resampler.process(&input);
        let steady = &out[400..out.len() - 400];
        // A sine's RMS is 1/sqrt(2); the filter must not change the level.
        assert!(
            (rms(steady) - rms(&input)).abs() < 0.05,
            "in {} vs out {}",
            rms(&input),
            rms(steady)
        );
    }

    #[test]
    fn a_tone_above_nyquist_is_rejected_rather_than_aliased() {
        // 7 kHz survives; 15 kHz would fold to 1 kHz at 16 kHz if unfiltered,
        // and Whisper would happily turn that whistle into words.
        let mut keep = Resampler::new(48_000);
        let kept = keep.process(&sine(48_000, 7_000.0, 0.5));

        let mut reject = Resampler::new(48_000);
        let rejected = reject.process(&sine(48_000, 15_000.0, 0.5));

        let kept_level = rms(&kept[400..kept.len() - 400]);
        let rejected_level = rms(&rejected[400..rejected.len() - 400]);
        assert!(kept_level > 0.5, "7 kHz should pass, got {kept_level}");
        assert!(
            rejected_level < kept_level / 50.0,
            "15 kHz should be rejected: kept {kept_level}, rejected {rejected_level}"
        );
    }

    #[test]
    fn block_size_does_not_change_the_result() {
        // The whole point of keeping filter state: a signal split across
        // callbacks must resample identically to the same signal in one go.
        let input = sine(48_000, 600.0, 0.4);

        let mut whole = Resampler::new(48_000);
        let one_shot = whole.process(&input);

        let mut streamed = Resampler::new(48_000);
        let mut pieces = Vec::new();
        // Deliberately ragged block sizes, as a real callback delivers.
        let mut offset = 0;
        for size in [512usize, 1024, 137, 4096, 999].iter().cycle() {
            if offset >= input.len() {
                break;
            }
            let end = (offset + *size).min(input.len());
            pieces.extend(streamed.process(&input[offset..end]));
            offset = end;
        }

        assert_eq!(pieces.len(), one_shot.len(), "sample count must match");
        for (index, (a, b)) in one_shot.iter().zip(pieces.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "sample {index} differs: {a} vs {b} — filter state is not continuous"
            );
        }
    }

    #[test]
    fn there_is_no_discontinuity_at_block_boundaries() {
        // The artefact the Python version's history buffer existed to avoid: a
        // step at each boundary shows up as a large sample-to-sample jump.
        let mut resampler = Resampler::new(48_000);
        let input = sine(48_000, 300.0, 0.3);
        let mut out = Vec::new();
        for chunk in input.chunks(480) {
            out.extend(resampler.process(chunk));
        }
        let steady = &out[200..out.len() - 200];
        let largest_step = steady
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .fold(0.0f32, f32::max);
        // A 300 Hz sine at 16 kHz moves at most ~0.12 per sample.
        assert!(largest_step < 0.2, "largest step {largest_step} suggests a boundary glitch");
    }

    #[test]
    fn silence_in_gives_silence_out() {
        let mut resampler = Resampler::new(44_100);
        let out = resampler.process(&vec![0.0f32; 44_100]);
        assert!(out.iter().all(|s| s.abs() < 1e-9));
    }

    #[test]
    fn an_empty_block_is_harmless() {
        let mut resampler = Resampler::new(48_000);
        assert!(resampler.process(&[]).is_empty());
    }

    #[test]
    fn an_awkward_rate_still_resamples() {
        // 44.1 kHz is the ratio that makes naive implementations fall over.
        let mut resampler = Resampler::new(44_100);
        let out = resampler.process(&sine(44_100, 440.0, 0.5));
        assert!(
            (out.len() as i64 - 8_000).abs() < 40,
            "got {} samples",
            out.len()
        );
        let candidates: Vec<f64> = (100..2000).step_by(10).map(|f| f as f64).collect();
        let found = dominant_freq(&out[600..], SAMPLE_RATE, &candidates);
        assert!((found - 440.0).abs() <= 10.0, "found {found} Hz");
    }
}
