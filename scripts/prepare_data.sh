#!/bin/bash
# prepare_data.sh
# Dataset preparation for PR-MaGIC.
#
# Usage (from PR-MaGIC root):
#   bash scripts/prepare_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_ROOT/data"

# ============================================================
# Datasets used by Matcher-IL
# (Instructions adapted from: https://github.com/aim-uofa/Matcher)
# ============================================================

echo "================================================================"
echo " Dataset Preparation for PR-MaGIC"
echo " Matcher dataset instructions from: github.com/aim-uofa/Matcher"
echo "================================================================"
echo ""

echo "--- 1. COCO-20i ---"
echo "Download images:"
echo "  wget http://images.cocodataset.org/zips/train2014.zip"
echo "  wget http://images.cocodataset.org/zips/val2014.zip"
echo "  wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
echo ""
echo "Download few-shot mask annotations from Google Drive (Matcher):"
echo "  train2014: https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge"
echo "  val2014:   https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1"
echo "  -> Place both under: data/COCO2014/annotations/"
echo ""
echo "Expected structure:"
echo "  data/COCO2014/"
echo "  ├── train2014/  ├── val2014/"
echo "  ├── annotations/train2014/  (mask PNGs)"
echo "  ├── annotations/val2014/"
echo "  └── splits/trn|val/fold{0-3}.pkl"
echo ""

echo "--- 2. FSS-1000 ---"
echo "Download from Google Drive (Matcher):"
echo "  https://drive.google.com/file/d/1Fn-cUESMMF1pQy8Xff-vPQvXJdZoUlP3"
echo ""
echo "Expected structure:"
echo "  data/FSS-1000/data/{class}/  splits/{trn,val,test}.txt"
echo ""

echo "--- 3. LVIS-92i ---"
echo "Download COCO2017 images:"
echo "  wget http://images.cocodataset.org/zips/train2017.zip"
echo "  wget http://images.cocodataset.org/zips/val2017.zip"
echo ""
echo "Download LVIS extended annotations from Google Drive (Matcher):"
echo "  lvis.zip: https://drive.google.com/file/d/1itJC119ikrZyjHB9yienUPD0iqV12_9y"
echo "  -> Extracts: data/LVIS/lvis_train.pkl  data/LVIS/lvis_val.pkl"
echo ""
echo "Expected structure:"
echo "  data/LVIS/coco/{train2017,val2017}/  lvis_train.pkl  lvis_val.pkl"
echo ""

echo "--- 4. PACO-Part ---"
echo "COCO2017 images are shared with LVIS."
echo ""
echo "Download PACO-Part annotations from Google Drive (Matcher):"
echo "  paco.zip: https://drive.google.com/file/d/1VEXgHlYmPVMTVYd8RkT6-l8GGq0G9vHX"
echo "  -> Extracts: data/PACO-Part/paco/paco_part_{train,val}.pkl"
echo ""
echo "Expected structure:"
echo "  data/PACO-Part/coco/{train2017,val2017}/  paco/paco_part_{train,val}.pkl"
echo ""

echo "--- 5. Pascal-Part ---"
echo "Download VOC2010 images:"
echo "  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
echo ""
echo "Download Pascal-Part annotations from Google Drive (Matcher):"
echo "  pascal.zip: https://drive.google.com/file/d/1WaM0VM6I9b3u3v3w-QzFLJI8d3NRumTK"
echo "  -> Extracts under: data/Pascal-Part/VOCdevkit/VOC2010/"
echo ""
echo "Expected structure:"
echo "  data/Pascal-Part/VOCdevkit/VOC2010/"
echo "  ├── JPEGImages/"
echo "  ├── Annotations_Part_json_merged_part_classes/"
echo "  └── all_obj_part_to_image.json"
echo ""

# ============================================================
# DIS5K (used by both PerSAM-IL and Matcher-IL)
# ============================================================

echo "--- 6. DIS5K ---"
echo "DIS5K requires accepting a terms-of-use agreement."
echo "Download from the official repository:"
echo "  https://github.com/xuebinqin/DIS  (see Data section)"
echo ""
echo "After downloading, extract so the root contains:"
echo "  DIS5K/Origin_Train/{im,gt}/  Origin_Test/{im,gt}/"
echo ""
echo "Then reorganize into class-subdirectory format:"
echo "  python scripts/prepare_dis5k.py --dis5k_root <path>/DIS5K"
echo ""
echo "Finally, symlink into data/:"
echo "  ln -s <path>/DIS5K $DATA_DIR/DIS5K"
echo ""
echo "Expected structure:"
echo "  data/DIS5K/Train/im/{1-1, 1-2, ..., 1-210}/"
echo "  data/DIS5K/Train/gt/  Test/im/  Test/gt/"
echo ""

echo "================================================================"
echo " All datasets ready. Verify symlinks in data/:"
echo "   ls -la $DATA_DIR"
echo "================================================================"
