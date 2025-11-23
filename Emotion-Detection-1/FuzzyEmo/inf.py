import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------
# CLASS LABELS (EDIT THIS TO MATCH YOUR TRAINING DATA ORDER)
# -----------------------------------------------------------
CLASS_LABELS = [
    "angry",     # 0
    "disgust",   # 1
    "fear",      # 2
    "happy",     # 3
    "sad",       # 4
    "surprise",  # 5
    "neutral",   # 6
    "contempt"   # 7
]
# If your training had a different order, update this list.


# -----------------------------------------------------------
# MODEL DEFINITIONS (your fuzzy architecture)
# -----------------------------------------------------------
class FuzzyConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding)
        self.alpha = nn.Parameter(torch.ones(out_ch) * 0.8)

    def forward(self, x):
        x = self.conv(x)
        return torch.tanh(self.alpha.view(1, -1, 1, 1) * x)

class FuzzyEmotionNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            FuzzyConv2D(3, 32, k=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            FuzzyConv2D(32, 64, k=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            FuzzyConv2D(64, 128, k=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
model = FuzzyEmotionNet(num_classes=8).to(device)
model.load_state_dict(torch.load("best.pth", map_location=device))
model.eval()


# -----------------------------------------------------------
# TRANSFORMS
# -----------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# -----------------------------------------------------------
# INFERENCE FUNCTION
# -----------------------------------------------------------
def infer(img_path, bbox=None):
    img = Image.open(img_path).convert("RGB")

    if bbox:
        x1, y1, x2, y2 = bbox
        img = img.crop((x1, y1, x2, y2))

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred = int(out.argmax(1).cpu().item())

    return pred, probs


# -----------------------------------------------------------
# RUN EXAMPLE
# -----------------------------------------------------------
if __name__ == "__main__":
    img_path = "./sample.png"    # change image here
    bbox = None                # or (x1,y1,x2,y2)

    pred, probs = infer(img_path, bbox=bbox)

    # Select labels array
    labels = CLASS_LABELS
    pred_label = labels[pred]

    print("Prediction index:", pred)
    print("Prediction emotion:", pred_label)

    print("\nTop-5 probabilities:")
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    for i, p in ranked[:5]:
        print(f"{labels[i]:<10s} : {p:.3f}")
