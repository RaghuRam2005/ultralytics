import torch
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

model = YOLO("yolov8n.pt")

def get_bn_weights(yolo_model):
    """
    Extracts all BatchNorm2d weights from a YOLO model and returns them as a single NumPy array.
    """
    bn_weights = []
    for module in yolo_model.model.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_weights.append(module.weight.detach().cpu().flatten())
    return torch.cat(bn_weights).numpy()

initial_bn_weights = get_bn_weights(model)

model.train(
    data="coco128.yaml",
    epochs=10,
    l1_lambda=0,
    freeze=0
)

final_bn_weights = get_bn_weights(model)

plt.figure(figsize=(12, 7))
plt.hist(initial_bn_weights, bins=100, color='royalblue', alpha=0.8)
plt.title('Distribution of BatchNorm Weights (Before Training)', fontsize=16)
plt.xlabel('Weight Value', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(12, 7))
plt.hist(final_bn_weights, bins=100, color='seagreen', alpha=0.8)
plt.title('Distribution of BatchNorm Weights (After Training with L1 Penalty)', fontsize=16)
plt.xlabel('Weight Value', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
