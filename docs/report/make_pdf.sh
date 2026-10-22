#!/bin/sh
# Convert report .docx files to PDF with LibreOffice, embedding the system Arial / Times New Roman fonts that the
# TÜBİTAK template uses (LibreOffice on macOS otherwise falls back to the metric-compatible Liberation fonts).
#   sh docs/report/make_pdf.sh docs/report/sonuc_raporu.docx [more.docx ...]
set -eu
SOFFICE=${SOFFICE:-/Applications/LibreOffice.app/Contents/MacOS/soffice}
FONTS=/System/Library/Fonts/Supplemental
PROFILE=$(mktemp -d)
trap 'rm -rf "$PROFILE"' EXIT
mkdir -p "$PROFILE/user/fonts"
for f in "$FONTS"/Arial*.ttf "$FONTS"/"Times New Roman"*.ttf; do cp "$f" "$PROFILE/user/fonts/"; done
for doc in "$@"; do
  "$SOFFICE" -env:UserInstallation="file://$PROFILE" --headless --convert-to pdf --outdir "$(dirname "$doc")" "$doc"
done
