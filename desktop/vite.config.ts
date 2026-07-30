import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri serves the dev server over a fixed port and fails rather than silently
// picking another one, so the Rust side always knows where the UI is.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      // src-tauri is Rust; cargo watches it. Vite reloading on .rs changes
      // would just thrash the browser.
      ignored: ["**/src-tauri/**"],
    },
  },
});
