//! Choosing where to cut a chunk.
//!
//! Port of `chunking.find_split_index`.

use crate::SAMPLE_RATE;

/// Granularity of the quiet-spot search.
const SPLIT_FRAME_SECONDS: f64 = 0.1;

/// Index of the quietest short frame at/after `search_from`.
///
/// Cutting there instead of at a fixed offset keeps chunk boundaries out of the
/// middle of words most of the time — which matters because the two sides of a
/// cut are transcribed independently, so a word split across it is lost from
/// both.
pub fn find_split_index(audio: &[f32], search_from: usize) -> usize {
    if search_from >= audio.len() {
        return audio.len();
    }

    let frame = ((SPLIT_FRAME_SECONDS * SAMPLE_RATE as f64) as usize).max(1);
    let window = &audio[search_from..];
    let frame_count = window.len() / frame;
    if frame_count == 0 {
        return audio.len();
    }

    // First (not last) minimum, matching numpy's argmin: with a long stretch of
    // true silence every frame ties at ~0, and cutting at the start of it keeps
    // the chunk shorter rather than dragging the boundary to the far end.
    let mut quietest = 0usize;
    let mut lowest = f32::INFINITY;
    for index in 0..frame_count {
        let start = index * frame;
        let energy =
            audio[search_from + start..search_from + start + frame]
                .iter()
                .map(|sample| sample.abs())
                .sum::<f32>()
                / frame as f32;
        if energy < lowest {
            lowest = energy;
            quietest = index;
        }
    }

    // Cut in the middle of the quiet frame, so neither side clips speech.
    search_from + quietest * frame + frame / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: usize = (0.1 * SAMPLE_RATE as f64) as usize; // 1600

    #[test]
    fn returns_len_when_search_starts_past_the_end() {
        let audio = vec![0.5f32; 100];
        assert_eq!(find_split_index(&audio, 100), 100);
        assert_eq!(find_split_index(&audio, 200), 100);
    }

    #[test]
    fn returns_len_when_less_than_one_frame_remains() {
        let audio = vec![0.5f32; FRAME + 10];
        assert_eq!(find_split_index(&audio, FRAME + 1), audio.len());
    }

    #[test]
    fn cuts_in_the_middle_of_the_quiet_frame() {
        // Loud everywhere except one frame, three frames past search_from.
        let mut audio = vec![0.5f32; FRAME * 10];
        let quiet = FRAME * 3;
        for sample in &mut audio[quiet..quiet + FRAME] {
            *sample = 0.0;
        }
        assert_eq!(find_split_index(&audio, 0), quiet + FRAME / 2);
    }

    #[test]
    fn searches_only_from_the_requested_offset() {
        // Quietest frame overall is at 0, but the search starts after it.
        let mut audio = vec![0.5f32; FRAME * 10];
        for sample in &mut audio[0..FRAME] {
            *sample = 0.0;
        }
        let quiet_later = FRAME * 6;
        for sample in &mut audio[quiet_later..quiet_later + FRAME] {
            *sample = 0.01;
        }
        let cut = find_split_index(&audio, FRAME * 4);
        assert_eq!(cut, quiet_later + FRAME / 2);
    }

    #[test]
    fn ties_pick_the_earliest_frame() {
        let audio = vec![0.0f32; FRAME * 5];
        assert_eq!(find_split_index(&audio, 0), FRAME / 2);
    }
}
