# predict.py

import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms

from model import create_model


def load_image(path):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = load_image(args.image).to(device)

    model = create_model(args.classes).to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    print("Predicted Class:", pred.item())
    print("\nThe predictied mood is :")
    if(pred.item() == 0):
        print("Anger")
    elif(pred.item() == 1):
        print("Contmpt")
    elif(pred.item() == 2):
        print("Disgust")
    elif(pred.item() == 3):
        print("Fear")
    elif(pred.item() == 4):
        print("Happy")
    elif(pred.item() == 5):
        print("Neutral")
    elif(pred.item() == 6):
        print("Sad")
    elif(pred.item() == 7):
        print("Surprise")
    else:
        print("ERROR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--classes", type=int, required=True)

    args = parser.parse_args()
    main(args)
