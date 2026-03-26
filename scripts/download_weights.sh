#!/bin/bash
# Download model weights into PR-MaGIC/weights/
#
# Usage (from PR-MaGIC root):
#   bash scripts/download_weights.sh           # download all
#   bash scripts/download_weights.sh --sam     # SAM only
#   bash scripts/download_weights.sh --dinov2  # DINOv2 only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="$(dirname "$SCRIPT_DIR")/weights"
mkdir -p "$WEIGHTS_DIR"

DL_SAM=false
DL_DINOV2=false

if [ $# -eq 0 ]; then
    DL_SAM=true
    DL_DINOV2=true
else
    for arg in "$@"; do
        case $arg in
            --sam)    DL_SAM=true ;;
            --dinov2) DL_DINOV2=true ;;
            *) echo "Unknown option: $arg"; exit 1 ;;
        esac
    done
fi

if [ "$DL_SAM" = true ]; then
    SAM_PATH="$WEIGHTS_DIR/sam_vit_h_4b8939.pth"
    if [ -f "$SAM_PATH" ]; then
        echo "[skip] sam_vit_h_4b8939.pth already exists."
    else
        echo "[download] SAM ViT-H weights..."
        wget -q --show-progress \
            https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
            -O "$SAM_PATH"
        echo "[done] $SAM_PATH"
    fi
fi

if [ "$DL_DINOV2" = true ]; then
    DINOV2_PATH="$WEIGHTS_DIR/dinov2_vitl14_pretrain.pth"
    if [ -f "$DINOV2_PATH" ]; then
        echo "[skip] dinov2_vitl14_pretrain.pth already exists."
    else
        echo "[download] DINOv2 ViT-L/14 weights..."
        wget -q --show-progress \
            https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth \
            -O "$DINOV2_PATH"
        echo "[done] $DINOV2_PATH"
    fi
fi

echo ""
echo "Weights directory: $WEIGHTS_DIR"
ls -lh "$WEIGHTS_DIR"
