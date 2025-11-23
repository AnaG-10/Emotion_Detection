# train.py

import argparse
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataset import AffectNetDataset
from model import create_model
from utils import train_one_epoch, evaluate


def get_transforms():
    """Return training and validation transforms."""
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_tf, val_tf


def count_classes(dataset):
    """Count unique classes in dataset."""
    unique = set()
    for _, lbl in dataset:
        unique.add(lbl)
    return len(unique)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load transforms
    train_tf, val_tf = get_transforms()

    # Load datasets
    print("\nLoading Dataset...")
    train_data = AffectNetDataset(
        os.path.join(args.dataset, "train/images"),
        os.path.join(args.dataset, "train/labels"),
        train_tf
    )
    val_data = AffectNetDataset(
        os.path.join(args.dataset, "val/images"),
        os.path.join(args.dataset, "val/labels"),
        val_tf
    )

    # Get number of classes
    num_classes = count_classes(train_data)
    print("Detected Classes:", num_classes)

    # Dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=2)

    # Model, Loss, Optimizer
    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Tracking Lists
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_acc = 0.0

    print("\nStarting Training...")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")
        print(f"Val Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_emotion_model1.pth")
            print("✔ Saved new best model")

    # Save training history
    np.savez(
        "training_history.npz",
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs
    )

    print("\nTraining Completed!")
    print("Best Validation Accuracy:", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to AffectNet root folder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
