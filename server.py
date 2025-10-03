from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# Config
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "./phishing_model_finetuned")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2048"))  # big batches are fast

# Pick best device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Make CPU go brrrr (no effect on MPS/CUDA)
try:
    if DEVICE.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass

app = FastAPI(title="ASU PhishGuard API")

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

# ----------------------------
# Input models
# ----------------------------
class Inp(BaseModel):
    text: str

class BatchInp(BaseModel):
    texts: List[str]

# ----------------------------
# Lazy-load model
# ----------------------------
_tokenizer = None
_model = None
_dtype_half_ok = False  # use half precision on MPS/CUDA

def get_model():
    global _tokenizer, _model, _dtype_half_ok
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    # Try dynamic quantization on CPU (helps a lot on Intel/AMD)
    if DEVICE.type == "cpu":
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception:
            # If quantization fails, continue with FP32 on CPU
            pass
        model.to(DEVICE)
    else:
        # Use half precision on GPU/MPS when possible
        _dtype_half_ok = True
        try:
            model.to(DEVICE, dtype=torch.float16)
        except Exception:
            _dtype_half_ok = False
            model.to(DEVICE)

    model.eval()
    _model = model
    return _tokenizer, _model

# ----------------------------
# Single prediction
# (keeps probabilities for your single-email UI bar)
# ----------------------------
@app.post("/predict")
def predict(inp: Inp):
    tokenizer, model = get_model()
    enc = tokenizer(inp.text, return_tensors="pt", truncation=True, max_length=512)
    # move to device (dtype chosen automatically by module weights)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.inference_mode():
        logits = model(**enc).logits[0]
        # cheap class pick
        pred = int(torch.argmax(logits).item())
        label = "malicious" if pred == 1 else "benign"
        # Only compute softmax for the single-email UI bar
        probs = torch.softmax(logits.float().cpu(), dim=-1).tolist()

    return {
        "label": label,
        "prob_malicious": float(probs[1]),
        "prob_benign": float(probs[0]),
        "threshold": THRESHOLD,
    }

# ----------------------------
# Batch prediction
# (fast path: argmax only; no softmax)
# ----------------------------
@app.post("/predict_batch")
def predict_batch(inp: BatchInp):
    tokenizer, model = get_model()
    texts = inp.texts
    results = []

    # Process large chunks to minimize overhead
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                        padding=True, max_length=512)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.inference_mode():
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).tolist()

        # Only return labels for speed (your batch UI doesn't need probs)
        for p in preds:
            label = "malicious" if int(p) == 1 else "benign"
            results.append({
                "label": label,
                # keep keys for backward-compat; set to None to avoid extra compute
                "prob_malicious": None,
                "prob_benign": None,
                "threshold": THRESHOLD,
            })

    return results
