//! Prints the chunk boundaries this crate would pick for an audio file.
//!
//! Exists to be diffed against `scripts/chunk_parity.py`, which prints the same
//! thing from the Python app's `chunking` module. Identical output means the
//! decode + split port is faithful, which is the precondition for an A/B
//! transcript comparison telling us anything about the *model*.
//!
//!     cargo run --example chunk_parity -- ../../data/some.mp3 4

use std::path::PathBuf;

use ai_meeting_lib::chunking::decode::{audio_duration, decode_range};
use ai_meeting_lib::chunking::split::find_split_index;
use ai_meeting_lib::SAMPLE_RATE;

const CHUNK_SECONDS: f64 = 300.0;
const SPLIT_SEARCH_SECONDS: f64 = 20.0;
const MIN_TAIL_SECONDS: f64 = 0.2;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let source = PathBuf::from(args.next().expect("usage: chunk_parity <audio> [max_chunks]"));
    let max_chunks: usize = args
        .next()
        .map(|value| value.parse().expect("max_chunks must be a number"))
        .unwrap_or(4);

    let duration = audio_duration(&source);
    println!("duration_sec={}", fmt_opt(duration));

    let mut start = 0.0f64;
    for index in 0..max_chunks {
        if let Some(total) = duration {
            if start >= total - MIN_TAIL_SECONDS {
                break;
            }
        }
        let audio = decode_range(&source, start, CHUNK_SECONDS + SPLIT_SEARCH_SECONDS)?;
        if audio.len() <= (MIN_TAIL_SECONDS * SAMPLE_RATE as f64) as usize {
            break;
        }

        let nominal_end = (CHUNK_SECONDS * SAMPLE_RATE as f64) as usize;
        let cut = if audio.len() > nominal_end {
            find_split_index(&audio, nominal_end).min(audio.len())
        } else {
            audio.len()
        };
        let next = start + cut as f64 / SAMPLE_RATE as f64;

        // The checksum catches a decode that lines up on length but not on
        // samples, which is exactly the failure a resampler difference causes.
        println!(
            "chunk={index} start={start:.3} decoded={} cut={cut} next={next:.3} sum={:.6}",
            audio.len(),
            audio[..cut].iter().map(|s| s.abs() as f64).sum::<f64>()
        );
        start = next;
    }
    Ok(())
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.3}")).unwrap_or_else(|| "none".into())
}
