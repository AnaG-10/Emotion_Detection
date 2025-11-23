from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

IMAGE_PATH = "IMG_6931.JPG"
NUM_CLASSES = 8

EMOTIONS = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fear",
    5: "Disgust",
    6: "Surprise",
    7: "Contempt"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -----------------------------
# Load WITHOUT pretrained weights
# -----------------------------
model = EfficientNet.from_name("efficientnet-b0")   # <-- FIXED
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)

model.load_state_dict(torch.load("efficientnet_best.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict_image(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        _, pred = torch.max(output, 1)

    class_id = pred.item()
    emotion = EMOTIONS.get(class_id, f"Class {class_id}")

    return class_id, emotion

class_id, emotion = predict_image(IMAGE_PATH)
print("Predicted Class ID:", class_id)
print("Predicted Emotion:", emotion)
