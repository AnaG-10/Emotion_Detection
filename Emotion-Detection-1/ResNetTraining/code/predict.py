from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms
import os
from model import create_model  

CONFIG = {
    "IMAGE_PATH": "sample.png",                  
    "WEIGHTS_PATH": "best_emotion_model.pth",    
    "NUM_CLASSES": 8,                            
    "LABELS_PATH": None                          
}


def load_image(path):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {path}")
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify/open image file: {path}")
    return tf(img).unsqueeze(0)


def load_weights_into_model(model, weights_path, device):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)

    # resolve common checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            # assume it's a state dict already
            state_dict = checkpoint
    else:
        # sometimes people save the whole model object (rare); try to handle gracefully
        try:
            model.load_state_dict(checkpoint.state_dict())
            model.to(device)
            model.eval()
            return model
        except Exception:
            # fallback: assume checkpoint *is* a state dict
            state_dict = checkpoint

    # remove 'module.' prefix if present (from DataParallel)
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    model.to(device)
    model.eval()
    return model


def load_labels(labels_path, num_classes):
    if labels_path:
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) != num_classes:
            raise ValueError(f"Labels count ({len(labels)}) doesn't match num_classes ({num_classes})")
        return labels

    # default labels for typical 8-class emotion dataset (FER-like)
    return ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def predict():
    image_path = CONFIG["IMAGE_PATH"]
    weights_path = CONFIG["WEIGHTS_PATH"]
    num_classes = CONFIG["NUM_CLASSES"]
    labels_path = CONFIG["LABELS_PATH"]

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load image
    print(f"Loading image: {image_path}")
    img = load_image(image_path).to(device)

    # create and load model
    print("Creating model...")
    model = create_model(num_classes)  # change if your create_model signature differs
    print(f"Loading weights from: {weights_path}")
    model = load_weights_into_model(model, weights_path, device)

    # forward pass
    with torch.no_grad():
        outputs = model(img)
        pred_idx = int(torch.argmax(outputs, dim=1).item())

    # labels
    labels = load_labels(labels_path, num_classes)
    predicted_label = labels[pred_idx] if 0 <= pred_idx < len(labels) else f"Class_{pred_idx}"

    # print results
    print(f"\nPredicted Class Index: {pred_idx}")
    print(f"Predicted Label: {predicted_label}")


if __name__ == "__main__":
    try:
        predict()
    except Exception as e:
        print("Error during prediction:")
        print(e)
