import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def build_model(num_classes, model_name="efficientnet-b0"):
    model = EfficientNet.from_pretrained(model_name)
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    return model
