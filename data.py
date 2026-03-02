import os, shutil, random
from pathlib import Path

DATA_ROOT = Path(r"C:\Users\ADMIN\Downloads\Train")
OUT_ROOT  = Path(r"C:\Users\ADMIN\Downloads\processed_split_2level")

SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
random.seed(SEED)

def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]

def split_files(files):
    files = files.copy()
    random.shuffle(files)
    n = len(files)
    n_train = int(n * SPLIT["train"])
    n_val   = int(n * SPLIT["val"])
    train = files[:n_train]
    val   = files[n_train:n_train+n_val]
    test  = files[n_train+n_val:]
    return train, val, test

# tạo cây thư mục output
for split_name in SPLIT:
    (OUT_ROOT / split_name).mkdir(parents=True, exist_ok=True)

total = 0
for fruit_dir in DATA_ROOT.iterdir():
    if not fruit_dir.is_dir():
        continue
    fruit = fruit_dir.name

    for quality in ["fresh", "rotten"]:
        q_dir = fruit_dir / quality
        if not q_dir.exists():
            continue

        files = list_images(q_dir)
        if not files:
            continue

        tr, va, te = split_files(files)
        for split_name, split_list in [("train", tr), ("val", va), ("test", te)]:
            dst_dir = OUT_ROOT / split_name / fruit / quality
            dst_dir.mkdir(parents=True, exist_ok=True)

            for i, src in enumerate(split_list):
                dst = dst_dir / f"{fruit}_{quality}_{i:06d}{src.suffix.lower()}"
                shutil.copy2(src, dst)

        total += len(files)
        print(f"{fruit}/{quality}: {len(files)} -> train {len(tr)}, val {len(va)}, test {len(te)}")

print("✅ Done. Total images:", total)
print("Output:", OUT_ROOT)