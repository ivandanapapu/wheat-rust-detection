import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.path.join("model", "wheat_rust_cnn.keras")
IMG_SIZE = (224, 224)

app = FastAPI(title="Wheat Rust Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("WARNING: Model file not found. Add model/wheat_rust_cnn.keras")

@app.get("/")
def home():
    return {"message": "Wheat Rust Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Train model and place it in backend/model/"}

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    probability = float(model.predict(arr, verbose=0)[0][0])
    label = "Rust Detected" if probability >= 0.5 else "Healthy Leaf"
    confidence = probability if probability >= 0.5 else 1 - probability

    recommendation = (
        "Rust signs found. Inspect crop area, isolate affected leaves, and consult an agriculture expert for treatment."
        if probability >= 0.5
        else "Leaf looks healthy. Continue regular monitoring and maintain good crop hygiene."
    )

    return {
        "label": label,
        "rust_probability": round(probability, 4),
        "confidence": round(confidence * 100, 2),
        "recommendation": recommendation
    }
