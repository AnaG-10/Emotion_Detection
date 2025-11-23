import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from model import build_model
import config
import os


def evaluate(dataset_path, model_name):

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
    print("Using device:", device)

    model = build_model(num_classes=len(test_data.classes), model_name=model_name)
    model.load_state_dict(torch.load(config.CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.numpy())
            pred_labels.extend(preds.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=test_data.classes))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test EfficientNet")
    parser.add_argument("--dataset", required=True, help="Dataset root containing test/")
    parser.add_argument("--model", default="efficientnet-b0", help="Model name")

    args = parser.parse_args()

    evaluate(args.dataset, model_name=args.model)
