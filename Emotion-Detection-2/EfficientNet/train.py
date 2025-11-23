import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

from model import build_model
import config


# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------
def get_loaders(dataset_path, batch_size):

    train_dir = os.path.join(dataset_path, "train")
    val_dir   = os.path.join(dataset_path, "val")

    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, train_tf)
    val_data   = datasets.ImageFolder(val_dir, val_tf)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_data.classes



# ------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------
def train_model(dataset_path, epochs, batch_size, model_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, classes = get_loaders(dataset_path, batch_size)
    num_classes = len(classes)
    print("Classes:", classes)

    model = build_model(num_classes, model_name=model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0

    # HISTORY ARRAYS
    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []



    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        model.train()
        correct, total = 0, 0
        running_loss = 0.0

        loop = tqdm(train_loader, leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Train Acc={train_acc:.4f}")

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------
        model.eval()
        val_correct, val_total = 0, 0
        val_running_loss = 0.0

        with torch.no_grad():
            val_loop = tqdm(val_loader, leave=False)
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_loop.set_postfix(loss=loss.item())


        val_loss = val_running_loss / val_total
        val_acc  = val_correct / val_total

        print(f"Val Acc={val_acc:.4f}")

        scheduler.step()

        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("💾 Saved Best Model")

        # Save last model every epoch
        torch.save(model.state_dict(), "last_model.pth")


    # ------------------------------------------------------------
    # SAVE TRAINING HISTORY
    # ------------------------------------------------------------
    np.savez("history.npz",
             train_losses=train_losses,
             train_accs=train_accs,
             val_losses=val_losses,
             val_accs=val_accs)

    print("\nSaved training history → history.npz")
    print("Training done!")



# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EfficientNet")

    parser.add_argument("--dataset", required=True, help="Dataset with train/ val/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model", type=str, default="efficientnet-b0")

    args = parser.parse_args()

    train_model(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model
    )
