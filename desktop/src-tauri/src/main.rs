// Keeps a console window from opening alongside the app on Windows release
// builds. Harmless on macOS, which is the primary target.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    ai_meeting_lib::run()
}
