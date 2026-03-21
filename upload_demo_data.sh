#!/usr/bin/env bash
# =============================================================================
# upload_demo_data.sh
#
# Uploads demo NIfTI files to GitHub Releases so the Gradio web apps can
# download them automatically when users click "Run demo".
#
# Usage:
#   1. Download the Dropbox folder:
#      https://www.dropbox.com/sh/9kmbytgf3jpj7bh/AACUZJ1KlJ1AFCPMIVyRFJi5a?dl=0
#      (click Download > Direct download)
#
#   2. Extract and place the demo files so this script can find them:
#      ./demo/ph_single_echo.nii
#      ./demo/mask_single_echo.nii
#      ./demo/ph_multi_echo.nii
#      ./demo/mag_multi_echo.nii
#      ./demo/mask_multi_echo.nii
#
#   3. Run this script:
#      bash upload_demo_data.sh
# =============================================================================

set -euo pipefail

DEMO_DIR="${1:-./demo}"   # pass path to demo folder as first arg, or default ./demo

# Compress .nii to .nii.gz if not already compressed
compress() {
    local src="$1"
    local dst="${src%.nii}.nii.gz"
    if [[ ! -f "$dst" ]]; then
        echo "Compressing $src -> $dst"
        gzip -k "$src"
    fi
    echo "$dst"
}

echo "=== Compressing demo files ==="
PH_SINGLE=$(compress "$DEMO_DIR/ph_single_echo.nii")
MASK_SINGLE=$(compress "$DEMO_DIR/mask_single_echo.nii")
PH_MULTI=$(compress "$DEMO_DIR/ph_multi_echo.nii")
MAG_MULTI=$(compress "$DEMO_DIR/mag_multi_echo.nii")
MASK_MULTI=$(compress "$DEMO_DIR/mask_multi_echo.nii")

echo ""
echo "=== Creating iQSM GitHub Release and uploading ==="
gh release create v1.0-demo \
    --repo sunhongfu/iQSM \
    --title "Demo data v1.0" \
    --notes "Single-echo in-vivo brain demo data for the iQSM web app.
Parameters: 1×1×1 mm, TE=20 ms, B0=3T." \
    "$PH_SINGLE" \
    "$MASK_SINGLE"

echo ""
echo "=== Creating iQSM_Plus GitHub Release and uploading ==="
gh release create v1.0-demo \
    --repo sunhongfu/iQSM_Plus \
    --title "Demo data v1.0" \
    --notes "Multi-echo in-vivo brain demo data for the iQSM+ web app.
Parameters: 1×1×1 mm, 8 echoes (TE=3.2–26.4 ms), B0=3T." \
    "$PH_MULTI" \
    "$MAG_MULTI" \
    "$MASK_MULTI"

echo ""
echo "=== Done! ==="
echo "Demo files are now hosted at:"
echo "  https://github.com/sunhongfu/iQSM/releases/tag/v1.0-demo"
echo "  https://github.com/sunhongfu/iQSM_Plus/releases/tag/v1.0-demo"
echo ""
echo "Users can now click 'Run demo' in the web app and it will download automatically."
