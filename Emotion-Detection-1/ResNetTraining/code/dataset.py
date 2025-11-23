# dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset


class AffectNetDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        self.img_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        txt_name = img_name.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(self.label_dir, txt_name)

        with open(label_path, "r") as f:
            class_id = int(f.readline().split()[0])

        if self.transform:
            image = self.transform(image)

        return image, class_id
