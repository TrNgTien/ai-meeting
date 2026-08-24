//! Map-reduce over a transcript that outgrows the model's context window.
//!
//! The context window is the constraint: a 60-minute meeting easily exceeds
//! 4096 tokens. So the transcript is split into pieces that each fit, each piece
//! is summarised on its own (the map), and the partial extractions are folded
//! into the final document (the reduce). A transcript short enough to fit in one
//! piece skips the reduce entirely.
//!
//! Output language follows [`crate::state::LanguageMode`]: Vietnamese headings
//! for `vi+en`/`auto`, English for `en`. The prompts themselves are the same
//! text in either language — the model reads them in its own language either way.

use crate::state::LanguageMode;

/// Whether headings and instructions are in Vietnamese.
fn vietnamese(language: LanguageMode) -> bool {
    language != LanguageMode::En
}

/// A transcript line that closes a finished file — the `----- Complete: … -----`
/// footer [`crate::chunking::transcribe_chunked`] writes. It is not speech, so
/// it has no place in a summary, exactly as `merge.rs` skips it.
fn is_footer_line(line: &str) -> bool {
    line.trim_start().starts_with("-----")
}

/// Split a transcript into chunks, each within `budget_chars`.
///
/// Splitting happens on line boundaries only — every transcript line is one
/// utterance, so this never splits mid-utterance. The footer is skipped, and a
/// single line longer than the budget is kept whole rather than torn in half
/// (better a long chunk than a truncated utterance).
pub fn split_for_map(text: &str, budget_chars: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        if is_footer_line(line) {
            continue;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        if !current.is_empty() && current.len() + trimmed.len() + 1 > budget_chars {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(trimmed);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// The map-stage system prompt: extract the useful bits from one excerpt.
pub fn map_system(language: LanguageMode) -> String {
    if vietnamese(language) {
        "Bạn là trợ lý tóm tắt cuộc họp chuyên nghiệp. Từ đoạn biên bản dưới đây, \
         hãy trích xuất: các quyết định đã thống nhất, các hành động cần làm (kèm \
         người phụ trách nếu có), và các câu hỏi còn bỏ ngỏ. Dùng tiếng Việt, viết \
         dạng gạch đầu dòng ngắn gọn. Nếu đoạn không có nội dung quan trọng, hãy trả \
         về \"(không có nội dung đáng chú ý)\"."
            .to_string()
    } else {
        "You are a professional meeting summarizer. From the transcript excerpt \
         below, extract: decisions reached, action items (with owners if named), \
         and open questions. Write in English, as concise bullet points. If the \
         excerpt has nothing notable, respond \"(nothing notable)\"."
            .to_string()
    }
}

/// The reduce-stage system prompt: fold the partial extractions into one document.
pub fn reduce_system(language: LanguageMode) -> String {
    if vietnamese(language) {
        "Bạn có các phần trích xuất từ nhiều đoạn của cùng một cuộc họp. Hãy gộp \
         chúng thành một tài liệu tóm tắt hoàn chỉnh gồm các mục: \"## Tóm tắt\", \
         \"## Quyết định\", \"## Hành động\", \"## Câu hỏi mở\". Gộp các mục trùng \
         nhau, giữ đầy đủ thông tin, viết bằng tiếng Việt."
            .to_string()
    } else {
        "You have extractions from several excerpts of the same meeting. Fold them \
         into one complete summary document with sections \"## Summary\", \
         \"## Decisions\", \"## Action items\", \"## Open questions\". Merge \
         duplicates, keep all information, write in English."
            .to_string()
    }
}

/// The user turn for the reduce stage: the partial extractions, numbered.
pub fn reduce_user(partials: &[String]) -> String {
    let mut out = String::new();
    for (index, partial) in partials.iter().enumerate() {
        out.push_str(&format!("--- Phần {}/{} ---\n", index + 1, partials.len()));
        out.push_str(partial);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript() -> String {
        let mut text = String::new();
        for i in 0..20 {
            text.push_str(&format!("[00:{:02}:00] Dòng nội dung số {i}\n", i));
        }
        text.push_str("----- Complete: 00:20:00 of audio · transcribed in 12s -----\n");
        text
    }

    #[test]
    fn chunks_never_exceed_the_budget() {
        let chunks = split_for_map(&transcript(), 80);
        assert!(chunks.len() > 1, "expected several chunks");
        for chunk in &chunks {
            assert!(chunk.len() <= 80, "chunk too big: {chunk:?}");
        }
    }

    #[test]
    fn splitting_never_breaks_a_line() {
        let chunks = split_for_map(&transcript(), 80);
        let all: String = chunks.join("\n");
        assert!(all.contains("Dòng nội dung số 19"), "no line may be lost");
        for chunk in &chunks {
            for line in chunk.lines() {
                assert!(line.starts_with('['), "a line was split: {line:?}");
            }
        }
    }

    #[test]
    fn the_footer_is_skipped() {
        let chunks = split_for_map(&transcript(), 10_000);
        let joined = chunks.join("\n");
        assert!(!joined.contains("Complete"), "footer leaked into the summary");
    }

    #[test]
    fn single_chunk_when_it_fits() {
        let chunks = split_for_map(&transcript(), 10_000);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn a_huge_single_line_is_kept_whole() {
        let line = format!("[00:00:00] {}", "x".repeat(5000));
        let chunks = split_for_map(&line, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), line.len());
    }

    #[test]
    fn reduce_user_numbers_the_partials() {
        let user = reduce_user(&["a".into(), "b".into()]);
        assert!(user.contains("Phần 1/2"));
        assert!(user.contains("Phần 2/2"));
    }

    #[test]
    fn language_switches_heading_language() {
        assert!(map_system(LanguageMode::ViEn).contains("Bạn là"));
        assert!(map_system(LanguageMode::En).contains("You are"));
        assert!(map_system(LanguageMode::Auto).contains("Bạn là"));
        assert!(reduce_system(LanguageMode::En).contains("## Summary"));
        assert!(reduce_system(LanguageMode::ViEn).contains("## Tóm tắt"));
    }
}
