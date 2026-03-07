"""
prepare_data.py — แบ่ง dataset เป็น train/val/test
โครงสร้าง folder ที่ต้องการ:
  data/raw/solar/      ← ภาพที่มีแผงโซลาร์
  data/raw/no_solar/   ← ภาพที่ไม่มีแผงโซลาร์
"""
import os
import shutil
import random
from pathlib import Path

RAW_DIR  = Path(r"C:\Users\sunny\Downloads\SolarScan_v2\SolarScan_v2\dataset\raw")
OUT_DIR  = Path("data/processed")
SPLITS   = {"train": 0.7, "val": 0.2, "test": 0.1}
CLASSES  = ["solar", "no-solar"]
SEED     = 42

random.seed(SEED)

def split_class(class_name: str):
    src = RAW_DIR / class_name
    images = list(src.glob("*.jpg")) + list(src.glob("*.png"))
    random.shuffle(images)

    n = len(images)
    train_end = int(n * SPLITS["train"])
    val_end   = train_end + int(n * SPLITS["val"])

    split_images = {
        "train": images[:train_end],
        "val":   images[train_end:val_end],
        "test":  images[val_end:]
    }

    for split, imgs in split_images.items():
        dest = OUT_DIR / split / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, dest / img.name)
        print(f"  {class_name}/{split}: {len(imgs)} images")

def main():
    print("🔄 Preparing dataset...\n")
    for cls in CLASSES:
        print(f"📁 Class: {cls}")
        split_class(cls)

    # สรุป
    print("\n✅ Done! Dataset structure:")
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            count = len(list((OUT_DIR / split / cls).glob("*")))
            print(f"  {split}/{cls}: {count} images")

if __name__ == "__main__":
    main()
