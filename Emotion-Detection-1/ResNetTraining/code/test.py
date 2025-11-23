# test.py

import argparse
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

from dataset import AffectNetDataset
from model import create_model
from utils import evaluate


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_data = AffectNetDataset(
        os.path.join(args.dataset, "test/images"),
        os.path.join(args.dataset, "test/labels"),
        tf
    )

    loader = DataLoader(test_data, batch_size=args.batch, shuffle=False)

    num_classes = len(set([lbl for _, lbl in test_data]))

    model = create_model(num_classes).to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    loss, acc = evaluate(model, loader, criterion, device)
    print("\nTest Loss:", loss)
    print("Test Accuracy:", acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--batch", type=int, default=32)

    args = parser.parse_args()
    main(args)
