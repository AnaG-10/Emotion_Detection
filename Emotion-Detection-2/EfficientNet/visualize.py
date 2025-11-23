import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import build_model
import config
import os


def visualize(dataset_path, model_name):

    # --------------------------------------------------------
    # LOAD TRAINING HISTORY
    # --------------------------------------------------------
    h = np.load("history.npz")

    train_losses = h["train_losses"]
    val_losses   = h["val_losses"]
    train_accs   = h["train_accs"]
    val_accs     = h["val_accs"]

    # --------------------------------------------------------
    # 1. PLOT ACCURACY
    # --------------------------------------------------------
    plt.figure(figsize=(10,5))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig("Acc.png")

    # --------------------------------------------------------
    # 2. PLOT LOSS
    # --------------------------------------------------------
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("Loss.png")


    # --------------------------------------------------------
    # 3. CONFUSION MATRIX USING TEST SET
    # --------------------------------------------------------
    test_dir = os.path.join(dataset_path, "test")

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    test_data = datasets.ImageFolder(test_dir, transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(num_classes=len(test_data.classes),
                        model_name=model_name)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            true_labels.extend(labels.numpy())
            pred_labels.extend(preds.cpu().numpy())


    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=test_data.classes,
                yticklabels=test_data.classes,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("Confusion.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="efficientnet-b0")
    
    args = parser.parse_args()
    
    visualize(args.dataset, args.model)
