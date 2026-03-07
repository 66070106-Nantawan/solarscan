"""
main.py — FastAPI Backend สำหรับ Solar Panel Classification
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os

app = FastAPI(title="SolarScan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── โหลด Model ตอน startup ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "model/weights/best.pt")
CLASSES = ["no_solar", "solar"]

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

model = load_model()

# ── Transform ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"message": "SolarScan API 🛰️"}

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ตรวจสอบ file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "รองรับเฉพาะ JPEG และ PNG")

    # อ่านภาพ
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Inference
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    solar_prob   = round(float(probs[1]), 4)   # prob ของ "มีแผง"
    no_solar_prob = round(float(probs[0]), 4)

    return {
        "prediction": CLASSES[pred_idx],
        "has_solar": pred_idx == 1,
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            "solar":    solar_prob,
            "no_solar": no_solar_prob
        }
    }
