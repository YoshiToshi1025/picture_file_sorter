#!/usr/bin/env python3
"""
Digital Camera Image Auto-Sorter

Organizes Nikon/Canon RAW and JPEG files into a dated folder hierarchy.

Destination structure:
    {dst}/{Maker}_{Model}/{YYYY}/{MM}/{DD}/
    {dst}/{Maker}_{Model}/{YYYY}/{MM}/{DD}/NKSC_PARAM/  (Nikon sidecar files)

Usage:
    python sorter.py --src <source_folder> --dst <destination_folder> [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    import exifread
except ImportError:
    print("Error: exifread is required. Install with: pip install exifread", file=sys.stderr)
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────

# All file extensions to scan (lowercase)
TARGET_EXTENSIONS: set[str] = {'.nef', '.cr2', '.cr3', '.jpg', '.jpeg'}

# Extensions that can only come from one maker
NIKON_ONLY: set[str] = {'.nef'}
CANON_ONLY: set[str] = {'.cr2', '.cr3'}

# Fragment of EXIF Make tag value → normalized maker folder name
MAKER_MAP: dict[str, str] = {
    'nikon': 'Nikon',
    'canon': 'Canon',
}

BRANCH_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Characters forbidden in Windows/macOS/Linux folder names, plus whitespace
_FORBIDDEN_RE = re.compile(r'[\\/:*?"<>|\s]+')


def sanitize_folder_name(name: str) -> str:
    """Replace forbidden characters and whitespace with '_', strip trailing dots/spaces."""
    sanitized = _FORBIDDEN_RE.sub('_', name.strip())
    return sanitized.strip('_') or '_'


def normalize_model(maker: str, model_str: str) -> str:
    """Strip redundant maker prefix from model string.

    e.g. maker='Nikon', model='NIKON Z 6'  →  'Z 6'
         maker='Nikon', model='Z 6'         →  'Z 6'  (unchanged)
    """
    prefix = maker.upper() + ' '
    if model_str.upper().startswith(prefix):
        model_str = model_str[len(prefix):]
    return model_str.strip()


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('sorter')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)-5s] %(message)s')

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ── EXIF ──────────────────────────────────────────────────────────────────────

def get_exif_info(file_path: Path) -> tuple[str | None, str | None, datetime | None]:
    """
    Extract maker name, camera folder name, and shooting datetime from EXIF.

    Returns (maker, camera_folder, shoot_dt) where:
        maker         – 'Nikon' or 'Canon' (used for logic checks)
        camera_folder – sanitized '{Maker}_{Model}' string (used as folder name)
        shoot_dt      – shooting datetime

    Returns (None, None, None) if EXIF is missing or maker is unrecognized.
    """
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        if not tags:
            return None, None, None

        make_tag = tags.get('Image Make')
        if not make_tag:
            return None, None, None

        maker: str | None = None
        for key, name in MAKER_MAP.items():
            if key in str(make_tag).strip().lower():
                maker = name
                break

        if not maker:
            return None, None, None

        model_tag = tags.get('Image Model')
        model_str = normalize_model(maker, str(model_tag).strip() if model_tag else '')

        # Build sanitized folder name: {Maker}_{Model}
        camera_folder = sanitize_folder_name(maker)
        if model_str:
            camera_folder = sanitize_folder_name(maker) + '_' + sanitize_folder_name(model_str)

        date_tag = (
            tags.get('EXIF DateTimeOriginal')
            or tags.get('EXIF DateTimeDigitized')
            or tags.get('Image DateTime')
        )
        if not date_tag:
            return None, None, None

        shoot_dt = datetime.strptime(str(date_tag).strip(), '%Y:%m:%d %H:%M:%S')
        return maker, camera_folder, shoot_dt

    except Exception:
        return None, None, None


# ── File utilities ────────────────────────────────────────────────────────────

def compute_md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def find_nksc(image_path: Path) -> Path | None:
    """Return the .nksc sidecar path if it exists, otherwise None.

    Convention: {image_dir}/NKSC_PARAM/{image_filename}.nksc
    e.g. DSC_0001.NEF  →  NKSC_PARAM/DSC_0001.NEF.nksc
    """
    nksc = image_path.parent / 'NKSC_PARAM' / (image_path.name + '.nksc')
    return nksc if nksc.exists() else None


def resolve_dest(dest_dir: Path, src_file: Path) -> tuple[Path, str]:
    """Resolve the final destination path for src_file inside dest_dir.

    Returns (dest_path, status) where status is:
        'new'      – no name conflict
        'dup'      – identical file already exists at dest → caller should skip
        'A'..'Z'   – branch suffix applied (e.g. 'A' means filename got '_A')

    Raises ValueError when all 26 branch suffixes are already taken.
    """
    stem = src_file.stem
    ext = src_file.suffix

    candidate = dest_dir / src_file.name
    if not candidate.exists():
        return candidate, 'new'
    if compute_md5(src_file) == compute_md5(candidate):
        return candidate, 'dup'

    for char in BRANCH_CHARS:
        candidate = dest_dir / f"{stem}_{char}{ext}"
        if not candidate.exists():
            return candidate, char
        if compute_md5(src_file) == compute_md5(candidate):
            return candidate, 'dup'

    raise ValueError(f"All branch suffixes (_A-_Z) exhausted for '{src_file.name}'")


def existing_ancestor(path: Path) -> Path:
    """Return the deepest existing ancestor of path (for disk_usage)."""
    p = path
    while not p.exists():
        p = p.parent
    return p


# ── Core logic ────────────────────────────────────────────────────────────────

def scan_candidates(src_root: Path) -> list[Path]:
    """Recursively collect all target image files, skipping NKSC_PARAM dirs."""
    return [
        p for p in src_root.rglob('*')
        if p.is_file()
        and p.suffix.lower() in TARGET_EXTENSIONS
        and 'NKSC_PARAM' not in p.parts
    ]


def build_tasks(
    candidates: list[Path],
    logger: logging.Logger,
) -> tuple[list[tuple[Path, str, str, datetime, Path | None]], int]:
    """
    Resolve EXIF for each candidate.
    Returns (task_list, skip_count).
    Each task is (img_path, maker, camera_folder, shoot_dt, nksc_path|None).
    """
    tasks: list[tuple[Path, str, str, datetime, Path | None]] = []
    skip_count = 0

    for img in candidates:
        ext = img.suffix.lower()
        maker, camera_folder, shoot_dt = get_exif_info(img)

        if not maker or not camera_folder or not shoot_dt:
            logger.warning(f"SKIP(no EXIF)       {img}")
            skip_count += 1
            continue

        # Reject impossible extension/maker combinations
        if ext in NIKON_ONLY and maker != 'Nikon':
            logger.warning(f"SKIP(maker mismatch) {img}")
            skip_count += 1
            continue
        if ext in CANON_ONLY and maker != 'Canon':
            logger.warning(f"SKIP(maker mismatch) {img}")
            skip_count += 1
            continue

        nksc = find_nksc(img) if maker == 'Nikon' else None
        tasks.append((img, maker, camera_folder, shoot_dt, nksc))

    return tasks, skip_count


def check_disk_space(
    tasks: list[tuple[Path, str, datetime, Path | None]],
    dst_root: Path,
    logger: logging.Logger,
) -> None:
    """Abort (sys.exit) if destination disk has insufficient free space."""
    total_bytes = sum(
        img.stat().st_size + (nksc.stat().st_size if nksc else 0)
        for img, _, _, _, nksc in tasks
    )
    required = int(total_bytes * 1.1)          # 10 % safety margin
    free = shutil.disk_usage(existing_ancestor(dst_root)).free

    logger.info(
        f"Size to move : {total_bytes / 1_048_576:.1f} MB"
        f"  (+10% margin: {required / 1_048_576:.1f} MB)"
        f"  |  Disk free: {free / 1_048_576:.1f} MB"
    )

    if free < required:
        logger.error(
            f"ABORTED: Insufficient disk space. "
            f"Need {required / 1_048_576:.1f} MB, "
            f"available {free / 1_048_576:.1f} MB."
        )
        sys.exit(1)


def do_move(src: Path, dst: Path, dry_run: bool, logger: logging.Logger) -> None:
    if dry_run:
        logger.info(f"  [DRY-RUN] {src}")
        logger.info(f"         -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def move_files(
    tasks: list[tuple[Path, str, str, datetime, Path | None]],
    dst_root: Path,
    dry_run: bool,
    logger: logging.Logger,
) -> dict[str, int]:
    stats: dict[str, int] = {'moved': 0, 'branch': 0, 'dup': 0, 'error': 0}

    for img, maker, camera_folder, shoot_dt, nksc in tasks:
        dest_dir = (
            dst_root
            / camera_folder
            / shoot_dt.strftime('%Y')
            / shoot_dt.strftime('%m')
            / shoot_dt.strftime('%d')
        )

        try:
            dest_img, status = resolve_dest(dest_dir, img)

            if status == 'dup':
                logger.info(f"SKIP(dup)           {img.name}")
                stats['dup'] += 1
                continue

            if status == 'new':
                logger.info(f"MOVE                {img}")
            else:
                logger.warning(f"MOVE(branch _{status})     {img}")
                logger.warning(f"                 -> {dest_img}")
                stats['branch'] += 1

            do_move(img, dest_img, dry_run, logger)
            stats['moved'] += 1

            # Move NKSC sidecar (destination name tracks resolved image name)
            if nksc:
                nksc_dest = dest_dir / 'NKSC_PARAM' / (dest_img.name + '.nksc')
                logger.info(f"MOVE(nksc)          {nksc}")
                do_move(nksc, nksc_dest, dry_run, logger)

        except ValueError as e:
            logger.error(f"ERROR {img.name}: {e}")
            stats['error'] += 1
        except OSError as e:
            logger.error(f"ERROR {img.name}: {e}")
            stats['error'] += 1

    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

def run(src_root: Path, dst_root: Path, dry_run: bool, logger: logging.Logger) -> None:
    # 1. Scan
    logger.info(f"Scanning : {src_root}")
    candidates = scan_candidates(src_root)
    logger.info(f"Candidates found: {len(candidates)}")

    # 2. EXIF resolve
    tasks, skip_exif = build_tasks(candidates, logger)
    logger.info(f"Processable: {len(tasks)}  /  EXIF-skip: {skip_exif}")

    if not tasks:
        logger.info("Nothing to move.")
        return

    # 3. Disk space check
    check_disk_space(tasks, dst_root, logger)

    # 4. Move
    stats = move_files(tasks, dst_root, dry_run, logger)

    # 5. Summary
    logger.info("=" * 60)
    logger.info(
        f"moved={stats['moved']}  branch={stats['branch']}  "
        f"dup-skip={stats['dup']}  exif-skip={skip_exif}  error={stats['error']}"
    )
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Auto-sort Nikon/Canon camera images into a dated folder hierarchy.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sorter.py --src D:/DCIM --dst E:/Photos
  python sorter.py --src D:/DCIM --dst E:/Photos --dry-run

Destination structure:
  {dst}/Nikon_D850/2024/03/15/DSC_0001.NEF
  {dst}/Nikon_D850/2024/03/15/NKSC_PARAM/DSC_0001.NEF.nksc
  {dst}/Canon_EOS_R5/2024/03/15/IMG_0001.CR3
        """,
    )
    parser.add_argument('--src', required=True, metavar='PATH', help='Source folder to scan (recursive)')
    parser.add_argument('--dst', required=True, metavar='PATH', help='Destination root folder')
    parser.add_argument('--dry-run', action='store_true', help='Preview only – no files are moved')
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    if not src_root.is_dir():
        print(f"Error: Source folder not found: {src_root}", file=sys.stderr)
        sys.exit(1)

    # Log file is written to the current working directory
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path.cwd() / f'sorter_{ts}.log'
    logger = setup_logging(log_path)

    if args.dry_run:
        logger.info("=== DRY-RUN MODE: no files will be moved ===")
    logger.info(f"Source : {src_root}")
    logger.info(f"Dest   : {dst_root}")
    logger.info(f"Log    : {log_path}")

    run(src_root, dst_root, args.dry_run, logger)


if __name__ == '__main__':
    main()
