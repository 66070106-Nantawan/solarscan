"""
evaluate.py — ประเมิน model บน test set
วัด Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
import json
import os

CONFIG = {
    "model_path": "model/weights/best.pt",
    "data_dir": "data/processed/test",
    "img_size": 224,
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def load_model(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load(path, map_location=CONFIG["device"]))
    model.eval()
    return model.to(CONFIG["device"])

def evaluate():
    transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(CONFIG["data_dir"], transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    classes = test_dataset.classes  # ['no_solar', 'solar']

    model = load_model(CONFIG["model_path"])

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(CONFIG["device"])
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of "solar"
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ── Metrics ──
    metrics = {
        "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds), 4),
        "recall":    round(recall_score(all_labels, all_preds), 4),
        "f1_score":  round(f1_score(all_labels, all_preds), 4),
        "auc_roc":   round(roc_auc_score(all_labels, all_probs), 4),
    }

    print("\n📊 Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("model/weights/confusion_matrix.png")
    print("✅ Saved confusion_matrix.png")

    # ── Log to MLflow ──
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("solarscan-classification")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("model/weights/confusion_matrix.png")
        with open("model/weights/eval_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact("model/weights/eval_results.json")

if __name__ == "__main__":
    evaluate()
