import os
import shutil
import random
from sklearn.model_selection import train_test_split

INPUT_ROOT = r"C:\Users\ADMIN\Downloads\Train"
OUTPUT_ROOT = r"C:\Users\ADMIN\Downloads\group23_22001611_22001624_split"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp")

random.seed(42)


fresh_files = []
rotten_files = []

for fruit in os.listdir(INPUT_ROOT):
    fruit_dir = os.path.join(INPUT_ROOT, fruit)
    if not os.path.isdir(fruit_dir):
        continue

    for quality in ["fresh", "rotten"]:
        q_dir = os.path.join(fruit_dir, quality)
        if not os.path.isdir(q_dir):
            continue

        for fn in os.listdir(q_dir):
            if not fn.lower().endswith(ALLOWED_EXT):
                continue

            full_path = os.path.join(q_dir, fn)

            if quality == "fresh":
                fresh_files.append(full_path)
            else:
                rotten_files.append(full_path)

print("Fresh:", len(fresh_files))
print("Rotten:", len(rotten_files))


def split_data(file_list):
    train, temp = train_test_split(
        file_list,
        test_size=(1 - TRAIN_RATIO),
        random_state=42
    )

    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42
    )

    return train, val, test

fresh_train, fresh_val, fresh_test = split_data(fresh_files)
rotten_train, rotten_val, rotten_test = split_data(rotten_files)


def create_dirs():
    for split in ["train", "val", "test"]:
        for label in ["fresh", "rotten"]:
            os.makedirs(os.path.join(OUTPUT_ROOT, split, label), exist_ok=True)

create_dirs()


def copy_files(file_list, split, label):
    for src in file_list:
        filename = os.path.basename(src)
        dst = os.path.join(OUTPUT_ROOT, split, label, filename)
        shutil.copy2(src, dst)

copy_files(fresh_train, "train", "fresh")
copy_files(fresh_val, "val", "fresh")
copy_files(fresh_test, "test", "fresh")

copy_files(rotten_train, "train", "rotten")
copy_files(rotten_val, "val", "rotten")
copy_files(rotten_test, "test", "rotten")

print("âœ… DONE. Dataset saved to:", OUTPUT_ROOT)
