import os
from pathlib import Path
import shutil
import random

# ----------------------------- #
# CONFIGURATION
# ----------------------------- #
source_dir = Path("cropped")                      # Folder containing class folders with images
target_dir = Path("cropped_split")                # Output folder for split dataset
train_ratio = 0.8                                 # 80% train, 20% validation

# Set random seed for reproducibility
random.seed(42)

# Create target directories for train and validation
for split in ['train', 'val']:
    split_path = target_dir / split
    split_path.mkdir(parents=True, exist_ok=True)

# Iterate over each species/class folder in the source directory
for class_dir in source_dir.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob("*.png"))    # List all PNG files in this class

        # Shuffle the images for a fair split
        random.shuffle(images)

        # Calculate split index
        split_idx = int(len(images) * train_ratio)

        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create class folders inside train and val directories
        train_class_dir = target_dir / 'train' / class_dir.name
        val_class_dir = target_dir / 'val' / class_dir.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # Copy training images
        for img in train_images:
            shutil.copy(img, train_class_dir / img.name)

        # Copy validation images
        for img in val_images:
            shutil.copy(img, val_class_dir / img.name)

        # Reporting
        print(f"{class_dir.name}: {len(train_images)} images in train, {len(val_images)} images in val")

print("Dataset splitting complete. Files are organized in 'cropped_split/train' and 'cropped_split/val'.")
