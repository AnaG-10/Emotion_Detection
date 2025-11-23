import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

from model import build_model
import config
import os


def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    # Load test dataset
    test_data = datasets.ImageFolder(config.TEST_DIR, transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    class_names = test_data.classes

    # Load trained model
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(config.CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []

    # RUN INFERENCE
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.numpy())
            pred_labels.extend(preds.cpu().numpy())

    # CLASSIFICATION REPORT
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    # OVERALL ACCURACY GRAPH
    acc = np.mean(np.array(true_labels) == np.array(pred_labels))

    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy"], [acc], color="skyblue")
    plt.ylim(0, 1)
    plt.title("Overall Test Accuracy")
    plt.ylabel("Accuracy")
    plt.savefig("test.png")


if __name__ == "__main__":
    test()
