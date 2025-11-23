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
    plt.show()


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

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Classification Report
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

    # Plot curves
    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs)

    # -----------------------------
    # Load model for confusion matrix
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detect number of classes
    val_images = f"{args.dataset}/val/images"
    val_labels = f"{args.dataset}/val/labels"

    # Temporary dataset to infer class count
    tmp_dataset = AffectNetDataset(val_images, val_labels)
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

    val_dataset = AffectNetDataset(val_images, val_labels, val_tf)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("\nComputing confusion matrix...")
    plot_confusion_matrix(model, val_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset folder containing train/val/test")
    parser.add_argument("--model", type=str, default="best_emotion_model.pth",
                        help="Path to model weights")
    parser.add_argument("--history", type=str, default="training_history.npz",
                        help="Path to training history file")

    args = parser.parse_args()
    main(args)
