//! Live recording of a meeting: the microphone and the machine's own output.
//!
//! Port of `audio_capture.py`.
//!
//! A meeting has two sides and the machine hears them on two different paths.
//! Your voice arrives through the microphone. Everyone else arrives as
//! *playback* — Zoom, Teams, Meet and friends decode the far end and send it to
//! the speakers, where an ordinary input device can't reach it.
//!
//! So the two sides are captured separately and kept separately:
//!
//! ```text
//! microphone  -> mic WAV     -> transcript labelled "Me"
//! system mix  -> system WAV  -> transcript labelled "Meeting"
//! ```
//!
//! Keeping them apart is what makes the merged transcript able to say who spoke,
//! and it also avoids the microphone's echo of the speakers being transcribed
//! twice. The cost is that the two streams are driven by independent clocks and
//! independent callbacks, which is what [`wav_writer`] exists to deal with.

pub mod devices;
pub mod mic;
pub mod recorder;
pub mod resampler;
pub mod system;
pub mod wav_writer;

/// Mean of all channels — every backend hands us interleaved frames and the
/// pipeline is mono all the way down.
pub fn to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_input_is_returned_unchanged() {
        let block = [0.1, 0.2, 0.3];
        assert_eq!(to_mono(&block, 1), block);
        assert_eq!(to_mono(&block, 0), block);
    }

    #[test]
    fn stereo_is_averaged_per_frame() {
        // L/R pairs: (1.0, 0.0) -> 0.5, (0.5, 0.5) -> 0.5, (-1.0, 1.0) -> 0.0
        let interleaved = [1.0, 0.0, 0.5, 0.5, -1.0, 1.0];
        assert_eq!(to_mono(&interleaved, 2), vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn a_ragged_tail_is_averaged_over_what_is_there() {
        // A backend can hand us a partial frame at the end of a stream; dividing
        // by the declared channel count would quietly attenuate it.
        let interleaved = [1.0, 1.0, 0.5];
        assert_eq!(to_mono(&interleaved, 2), vec![1.0, 0.5]);
    }
}
