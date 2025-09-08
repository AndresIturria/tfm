#!/usr/bin/env python3
import argparse, os, random, math, shutil
from pathlib import Path
from typing import Iterable, List, Set

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}

def list_class_dirs(split_dir: Path) -> List[str]:
    if not split_dir.exists():
        return []
    return sorted([p.name for p in split_dir.iterdir() if p.is_dir()])

def list_images(class_dir: Path) -> List[Path]:
    return [p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_one(src: Path, dst: Path, mode: str):
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def choose_classes(train_root: Path, val_root: Path, k: int, seed: int, strategy: str, allow_missing: bool=False) -> List[str]:
    train_classes = set(list_class_dirs(train_root))
    val_classes   = set(list_class_dirs(val_root))
    both = train_classes & val_classes

    if not both:
        raise SystemExit("No common classes found between training and validation.")

    if len(both) < k:
        msg = f"Only {len(both)} classes are common to both splits; cannot choose {k}."
        if allow_missing:
            print("[WARN]", msg, "Proceeding with all common classes.")
            k = len(both)
        else:
            raise SystemExit(msg)

    pool = sorted(both)
    if strategy == "random":
        rng = random.Random(seed)
        rng.shuffle(pool)
        chosen = pool[:k]
    elif strategy == "lexicographic":
        chosen = pool[:k]
    else:
        raise ValueError("strategy must be 'random' or 'lexicographic'")
    return sorted(chosen)

def copy_split(src_split: Path, dst_split: Path, classes: List[str], frac: float, seed: int, mode: str):
    total_src = total_take = total_done = 0
    rng = random.Random(seed)
    ensure_dir(dst_split)

    for cls in classes:
        src_cls = src_split / cls
        dst_cls = dst_split / cls
        ensure_dir(dst_cls)

        files = list_images(src_cls)
        n = len(files)
        if n == 0:
            print(f"[{src_split.name}] {cls}: 0 files (skipped)")
            continue

        if frac >= 1.0:
            take = files
            k = n
        else:
            k = max(1, math.ceil(frac * n))
            rng.shuffle(files)
            take = files[:k]

        for f in take:
            try:
                copy_one(f, dst_cls / f.name, mode)
                total_done += 1
            except Exception as e:
                print(f"[WARN] Could not copy {f} -> {dst_cls/f.name}: {e}")

        total_src += n
        total_take += k
        print(f"[{src_split.name}] {cls}: {k}/{n} selected")

    print(f"[{src_split.name}] Selected {total_take} of {total_src} files; {total_done} placed.")

def main():
    ap = argparse.ArgumentParser(description="Copy a subset of classes (and optionally a fraction of images) from an ImageNet-style tree.")
    ap.add_argument("--src", default="data-imagenet", type=Path,
                    help="Source root containing 'training' and 'validation'")
    ap.add_argument("--dst", default="datacopy", type=Path,
                    help="Destination root")
    ap.add_argument("--num-classes", "-k", type=int, default=50,
                    help="Number of classes to keep")
    ap.add_argument("--strategy", choices=["random", "lexicographic"], default="random",
                    help="How to choose classes")
    ap.add_argument("--fraction", "-f", type=float, default=1.0,
                    help="Fraction of images per selected class (e.g., 0.10 for 10%%)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed (for class & image selection)")
    ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy",
                    help="Copy files or create links to save space")
    ap.add_argument("--allow-missing", action="store_true",
                    help="If fewer than K common classes exist, proceed with all common classes")
    args = ap.parse_args()

    train_src = args.src / "training"
    val_src   = args.src / "validation"
    train_dst = args.dst / "training"
    val_dst   = args.dst / "validation"

    if not train_src.exists() or not val_src.exists():
        raise SystemExit(f"Expected 'training' and 'validation' under {args.src}")

    chosen = choose_classes(train_src, val_src, args.num_classes, args.seed, args.strategy, allow_missing=args.allow_missing)
    print(f"Chosen classes ({len(chosen)}): {', '.join(chosen[:10])}{' ...' if len(chosen)>10 else ''}")

    # Make root dirs
    ensure_dir(train_dst); ensure_dir(val_dst)

    print(f"Source: {args.src}  ->  Destination: {args.dst}  (mode={args.mode})")
    print(f"Classes: {len(chosen)}  |  Fraction per class: {args.fraction*100:.2f}%  |  Seed: {args.seed}")

    # Copy both splits using the SAME class list
    copy_split(train_src, train_dst, chosen, args.fraction, args.seed,     args.mode)
    copy_split(val_src,   val_dst,   chosen, args.fraction, args.seed + 1, args.mode)

    print("Done.")

if __name__ == "__main__":
    main()
