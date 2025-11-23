# visualise.py

import argparse
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report

from dataset import AffectNetDataset
from model import create_model


# ---------------------------------------------------------------------
# 1. Plot Training Loss & Accuracy
# ---------------------------------------------------------------------
def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(14, 6))

    # ---- LOSS ----
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # ---- ACCURACY ----
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy", linewidth=2)
    plt.plot(val_accs, label="Val Accuracy", linewidth=2)
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("2.png")


# ---------------------------------------------------------------------
# 2. Confusion Matrix
# ---------------------------------------------------------------------
def plot_confusion_matrix(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("1.png")

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4))


# ---------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------
def main(args):

    # -----------------------------
    # Load training history
    # -----------------------------
    print("Loading training history:", args.history)

    history = np.load(args.history)
    train_losses = history["train_losses"]
    val_losses = history["val_losses"]
    train_accs = history["train_accs"]
    val_accs = history["val_accs"]

    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs)

    # -----------------------------
    # Load model
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validation folder path (corrected)
    val_dir = f"{args.dataset}/val"

    # Infer number of classes
    tmp_dataset = AffectNetDataset(val_dir)
    num_classes = len(set([lbl for _, lbl in tmp_dataset]))
    
    print("Detected Classes:", num_classes)

    model = create_model(num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # -----------------------------
    # Create validation loader
    # -----------------------------
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_dataset = AffectNetDataset(val_dir, val_tf)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("\nComputing confusion matrix...")
    plot_confusion_matrix(model, val_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset containing train/val/test folders")
    parser.add_argument("--model", type=str, default="best_emotion_model.pth",
                        help="Path to model weights")
    parser.add_argument("--history", type=str, default="training_history.npz",
                        help="Path to training history")

    args = parser.parse_args()
    main(args)
