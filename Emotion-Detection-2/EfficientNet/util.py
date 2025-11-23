import os
import shutil
import random

def create_validation_split(
        dataset_path,
        val_ratio=0.2,
        seed=42
    ):
    random.seed(seed)

    train_dir = os.path.join(dataset_path, "train")
    val_dir   = os.path.join(dataset_path, "val")

    # Create val directory if not exists
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        print(f"Created {val_dir}")

    # Iterate over each class folder
    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)

        # Skip non-directories
        if not os.path.isdir(class_train_path):
            continue

        # Create corresponding class folder inside val/
        class_val_path = os.path.join(val_dir, class_name)
        os.makedirs(class_val_path, exist_ok=True)

        # List all images in train/class
        images = [
            img for img in os.listdir(class_train_path)
            if img.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        # How many to move
        val_count = int(len(images) * val_ratio)
        print(f"{class_name}: Moving {val_count} images to validation set...")

        # Randomly pick images
        val_images = random.sample(images, val_count)

        # Move images
        for img in val_images:
            src = os.path.join(class_train_path, img)
            dst = os.path.join(class_val_path, img)
            shutil.move(src, dst)

    print("\nValidation dataset created successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create validation split")
    parser.add_argument("--dataset", required=True, help="Dataset root directory")
    parser.add_argument("--ratio", type=float, default=0.2, help="Validation ratio")

    args = parser.parse_args()

    create_validation_split(args.dataset, val_ratio=args.ratio)
