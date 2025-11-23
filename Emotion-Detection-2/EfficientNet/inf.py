import torch
from torchvision import transforms
from PIL import Image
import json
import config
from model import build_model


def load_class_names():
    with open("classes.json", "r") as f:
        return json.load(f)


def predict(image_path, model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load class names
    classes = load_class_names()
    num_classes = len(classes)

    # Load model (EfficientNet-B0 fixed)
    model = build_model(num_classes, model_name="efficientnet-b0")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    class_name = classes[predicted.item()]
    conf = confidence.item()

    print(f"\nImage: {image_path}")
    print(f"Predicted Class: {class_name}")
    print(f"Confidence: {conf:.4f}\n")

    return class_name, conf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EfficientNet-B0 Image Inference")

    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--model", required=True,
                        help="Path to model weights (.pth) — best or last")

    args = parser.parse_args()

    predict(args.image, args.model)
