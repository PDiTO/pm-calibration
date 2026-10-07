#!/usr/bin/env bash
# Render report.html to pm-calibration-report.pdf with headless Chrome, then set the PDF's
# title and author. Needs Chrome or Chromium 131 or later (for the page footers) and uv.
# The figures are the SVGs that `pmcal report` writes to results/figures.
set -euo pipefail
cd "$(dirname "$0")"

chrome="${CHROME:-$(command -v google-chrome || command -v chromium || command -v chromium-browser)}"
out="pm-calibration-report.pdf"
tmp="$(mktemp --suffix=.pdf)"
trap 'rm -f "$tmp"' EXIT

"$chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=10000 --print-to-pdf="$tmp" "file://$PWD/report.html" 2>/dev/null

uv run --quiet --no-project --with pypdf python - "$tmp" "$out" <<'EOF'
import sys

from pypdf import PdfReader, PdfWriter

src, dst = sys.argv[1:]
pdf = PdfWriter(clone_from=PdfReader(src))
pdf.add_metadata(
    {
        "/Title": "Are prediction market prices calibrated?",
        "/Subject": "Polymarket and Kalshi, January to September 2026",
        "/Author": "Paul Davis",
    }
)
with open(dst, "wb") as f:
    pdf.write(f)
print(f"wrote {dst}: {len(pdf.pages)} pages")
EOF
