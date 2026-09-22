"""FastAPI service that serves the Food-11 classifier from the MLflow registry.

Endpoints
---------
GET  /health   -> {"status": "ok", ...}
POST /predict  -> multipart upload of one image; returns the predicted category
                  and a confidence score.

Design notes
------------
The model is loaded **once at startup** through an MLflow model URI
(`models:/food11@champion`), never from a `.pth` path. That indirection is the
whole point: the container is built without knowing which model version it will
serve, and promoting a new version is a registry operation, not a rebuild.

The tracking URI comes from the environment so the same image works locally
(`http://127.0.0.1:5000`) and inside a container
(`http://host.docker.internal:5000`), with no code change.

Usage
-----
    uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
import os

import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("food11.serve")

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = os.environ.get("MODEL_URI", "models:/food11@champion")

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "128"))

# Must match the preprocessing used in train.py, or the model sees inputs
# distributed differently from anything it was trained on.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# torchvision's ImageFolder assigns label indices by sorting the class directory
# names, so this list must stay in sorted order to match the training labels.
# The model itself carries no class names — only 11 output logits.
CLASS_NAMES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

# Populated at startup by the handler below.
state: dict[str, object] = {}


def load_model():
    """Resolve the alias and download the model through the tracking server."""
    mlflow.set_tracking_uri(TRACKING_URI)
    logger.info("tracking uri: %s", TRACKING_URI)
    logger.info("loading model: %s", MODEL_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("model loaded")
    return model


app = FastAPI(
    title="Food-11 classifier",
    description="Serves the model registered as food11@champion in MLflow.",
    version="1.0.0",
)


@app.on_event("startup")
def startup() -> None:
    # Loading here rather than per-request means the (slow) download and
    # deserialisation happen once, and a bad model URI fails loudly at boot
    # instead of on the first user request.
    state["model"] = load_model()


def preprocess(raw: bytes) -> np.ndarray:
    """Bytes of an uploaded image -> a (1, 3, H, W) float32 batch."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:  # noqa: BLE001 - any decode failure is a 400
        raise HTTPException(status_code=400,
                            detail=f"could not read image: {exc}") from exc

    array = (array - IMAGENET_MEAN) / IMAGENET_STD     # H, W, C
    array = array.transpose(2, 0, 1)                   # C, H, W
    return array[np.newaxis, ...].astype(np.float32)   # 1, C, H, W


def softmax(logits: np.ndarray) -> np.ndarray:
    """Turn raw logits into probabilities, shifted for numerical stability."""
    shifted = logits - logits.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "tracking_uri": TRACKING_URI,
        "model_loaded": "model" in state,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    batch = preprocess(raw)
    logits = np.asarray(model.predict(batch)).reshape(-1)
    probabilities = softmax(logits)
    best = int(probabilities.argmax())

    return {
        "filename": file.filename,
        "category": CLASS_NAMES[best],
        "category_index": best,
        "confidence": round(float(probabilities[best]), 4),
        "probabilities": {
            name: round(float(probability), 4)
            for name, probability in zip(CLASS_NAMES, probabilities)
        },
    }