# Phase 1 Presentation Sources & Export

This directory contains the editable presentation slides and interactive web export for Phase 1 of the Smart Waste Segregation project.

## Structure
- `slides.md`: Markdown presentation source formatted for Marp, Slidev, or GitHub rendering.
- `index.html`: Interactive, standalone, zero-dependency HTML presentation viewer with keyboard navigation, slide transitions, and printable stylesheet.
- `figures/`: Key visual artifacts embedded in the presentation.

## Viewing & Presenting
Open `index.html` in any modern web browser:
- **Next Slide**: Right Arrow, Spacebar, Page Down, or the "Next" button.
- **Previous Slide**: Left Arrow, Page Up, or the "Previous" button.
- **First / Last Slide**: Home / End keys.
- **Fullscreen**: Click the "Fullscreen" button or press F11.

## Exporting to PDF
1. Open `index.html` in Chrome or any Chromium-based browser.
2. Press `Cmd + P` (macOS) or `Ctrl + P` (Linux/Windows).
3. Set Destination to "Save as PDF", Layout to "Landscape", Margins to "None", and enable "Background graphics".
4. Alternatively, use Marp CLI:
   ```bash
   npx @marp-team/marp-cli slides.md --pdf -o presentation.pdf
   ```
