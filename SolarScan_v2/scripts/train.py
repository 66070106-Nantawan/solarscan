"""
train.py — Fine-tune ResNet50 สำหรับ Solar Panel Classification
Binary Classification: มีแผงโซลาร์ (1) / ไม่มีแผงโซลาร์ (0)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import mlflow
import mlflow.pytorch
from datetime import datetime
import os

# ========== Config ==========
CONFIG = {
    "model_name": "resnet50",
    "epochs": 20,
    "batch_size": 32,
    "lr": 0.001,
    "img_size": 224,
    "num_classes": 2,
    "data_dir": "data/processed",
    "save_path": "model/weights/best.pt",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ========== Data Transforms ==========
train_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225])    # ImageNet std
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def build_model():
    """โหลด ResNet50 pretrained แล้วเปลี่ยน head เป็น 2 classes"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze layers ส่วนใหญ่ — train แค่ layer สุดท้าย
    for param in model.parameters():
        param.requires_grad = False

    # เปลี่ยน classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, CONFIG["num_classes"])
    )
    return model

def train():
    print(f"🚀 Training on: {CONFIG['device']}")

    # โหลด dataset
    train_dataset = datasets.ImageFolder(
        os.path.join(CONFIG["data_dir"], "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(CONFIG["data_dir"], "val"),
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False)

    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"📌 Classes: {train_dataset.classes}")  # ['no-solar', 'solar']

    # Model, Loss, Optimizer
    model = build_model().to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("solarscan-classification")

    best_val_acc = 0.0

    with mlflow.start_run(run_name=f"resnet50_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_params(CONFIG)

        for epoch in range(CONFIG["epochs"]):
            # ── Train ──
            model.train()
            train_loss, train_correct = 0, 0

            for images, labels in train_loader:
                images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()

            train_acc = train_correct / len(train_dataset)

            # ── Validate ──
            model.eval()
            val_loss, val_correct = 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            val_acc = val_correct / len(val_dataset)
            scheduler.step()

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss / len(train_loader),
                "train_acc": train_acc,
                "val_loss": val_loss / len(val_loader),
                "val_acc": val_acc,
            }, step=epoch)

            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            # บันทึก best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), CONFIG["save_path"])
                print(f"  ✅ Saved best model (val_acc={val_acc:.4f})")

        mlflow.log_metric("best_val_acc", best_val_acc)
        print(f"\n🏆 Best Val Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()
