"""
prepare_dis5k.py
Reorganize DIS5K from flat-file format into class-subdirectory format
required by the data loader.

Original format (Origin_Train/im/, Origin_Test/im/):
    {cat_id}#{cat_name}#{subcat_id}#{subcat_name}#{image_name}.jpg
    e.g.  1#Accessories#1#Bag#some_photo.jpg

Target format (Train/im/, Test/im/):
    {cat_id}-{subcat_id}/{image_name}.jpg
    e.g.  1-1/some_photo.jpg

GT masks follow the same structure with .png extension.

Usage (from PR-MaGIC root):
    python scripts/prepare_dis5k.py --dis5k_root data/DIS5K

    # Dry run (shows what would be created without copying):
    python scripts/prepare_dis5k.py --dis5k_root data/DIS5K --dry_run
"""

import os
import shutil
import argparse
from pathlib import Path


def parse_class_dir(filename: str) -> str:
    """
    Extract class directory name from a DIS5K filename.
    Format: {cat_id}#{cat_name}#{subcat_id}#{subcat_name}#{image_name}.ext
    Returns: '{cat_id}-{subcat_id}'
    """
    parts = filename.split('#')
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")
    return f"{parts[0]}-{parts[2]}"


def reorganize_split(origin_dir: Path, target_dir: Path, ext_filter: str, dry_run: bool):
    """
    Reorganize files from origin_dir (flat) into target_dir (class subdirs).
    ext_filter: file extension to process, e.g. '.jpg' or '.png'
    """
    if not origin_dir.exists():
        print(f"[skip] {origin_dir} does not exist.")
        return

    files = sorted(f for f in origin_dir.iterdir()
                   if f.is_file() and f.suffix.lower() == ext_filter)

    moved, skipped = 0, 0
    for f in files:
        try:
            class_dir_name = parse_class_dir(f.name)
        except ValueError as e:
            print(f"  [warn] {e}")
            skipped += 1
            continue

        dest_dir = target_dir / class_dir_name
        dest_file = dest_dir / f.name

        if dest_file.exists():
            skipped += 1
            continue

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest_file)
        else:
            print(f"  [dry] {f.name} -> {dest_dir.name}/{f.name}")
        moved += 1

    print(f"  {'[dry]' if dry_run else '[done]'} {moved} files processed, {skipped} skipped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dis5k_root', type=str, required=True,
                        help='Path to DIS5K root directory (contains Origin_Train/, Origin_Test/)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print actions without copying files')
    args = parser.parse_args()

    root = Path(args.dis5k_root)

    splits = [
        ('Origin_Train', 'Train'),
        ('Origin_Test',  'Test'),
    ]

    for origin_name, target_name in splits:
        for subdir, ext in [('im', '.jpg'), ('gt', '.png')]:
            origin = root / origin_name / subdir
            target = root / target_name / subdir
            print(f"\n[{origin_name}/{subdir}] -> [{target_name}/{subdir}]")
            reorganize_split(origin, target, ext, args.dry_run)

    if not args.dry_run:
        print("\nDone. Verify with:")
        print(f"  ls {root}/Train/im/ | wc -l   # expect 210 class dirs")
        print(f"  ls {root}/Test/im/  | wc -l   # expect 210 class dirs")


if __name__ == '__main__':
    main()
