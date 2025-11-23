import os
from PIL import Image
from torch.utils.data import Dataset




class AffectNetDataset(Dataset):
"""Simple dataset that expects image files and paired .txt labels


images: path to directory containing image files (.jpg/.png/.jpeg)
labels: path to directory containing label files named like <image-stem>.txt
transform: torchvision transforms
"""


def __init__(self, images, labels, transform=None):
self.images = images
self.labels = labels
self.transform = transform


# list only image files
self.files = [
f for f in os.listdir(images)
if f.lower().endswith((".jpg", ".png", ".jpeg"))
]
self.files.sort()


def __len__(self):
return len(self.files)


def __getitem__(self, idx):
img_name = self.files[idx]
img_path = os.path.join(self.images, img_name)
image = Image.open(img_path).convert("RGB")


label_name = img_name.rsplit(".", 1)[0] + ".txt"
label_path = os.path.join(self.labels, label_name)


# default label if missing -> raise to catch dataset errors early
if not os.path.exists(label_path):
raise FileNotFoundError(f"Label file not found: {label_path}")


with open(label_path, "r") as f:
class_id = int(f.readline().split()[0])


if self.transform:
image = self.transform(image)


return image, class_id