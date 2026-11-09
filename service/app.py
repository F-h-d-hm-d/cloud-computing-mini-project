
import json
import os

import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/v2")
FIELDS = ("intent", "date", "time", "place")

model = None
tokenizer = None
load_error = None


# -----------------------------
# Load trained student model
# -----------------------------

def load_model():
    global model, tokenizer, load_error

    try:
        torch.set_num_threads(4)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

        model.to("cpu")
        model.eval()

        load_error = None

        print("Model loaded:", MODEL_PATH, flush=True)

    except Exception as exc:
        load_error = str(exc)
        print("Model loading failed:", load_error, flush=True)


# -----------------------------
# Command parsing inference
# -----------------------------

def parse_command(text):
    prompt = f"Command: {text}\nJSON: "

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    raw = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    result = json.loads(raw)

    if not isinstance(result, dict) or set(result) != set(FIELDS):
        raise ValueError("Model returned an invalid JSON structure")

    if not isinstance(result["intent"], str):
        raise ValueError("Invalid intent")

    for field in ("date", "time", "place"):
        if result[field] is not None and not isinstance(result[field], str):
            raise ValueError(f"Invalid {field}")

    return result


# -----------------------------
# Kubernetes liveness probe
# -----------------------------

@app.get("/health")
def health():
    return jsonify({
        "status": "alive"
    }), 200


# -----------------------------
# Kubernetes readiness probe
# -----------------------------

@app.get("/ready")
def ready():
    if model is None:
        return jsonify({
            "status": "not_ready",
            "error": load_error
        }), 503

    return jsonify({
        "status": "ready"
    }), 200


# -----------------------------
# Main inference API
# -----------------------------

@app.post("/parse")
def parse():
    if model is None:
        return jsonify({
            "error": "Model not ready"
        }), 503

    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify({
            "error": "Expected JSON object"
        }), 400

    text = body.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "Provide non-empty text"
        }), 400

    try:
        result = parse_command(text)
        return jsonify(result), 200

    except Exception as exc:
        app.logger.exception("Inference failed")

        return jsonify({
            "error": "Inference failed",
            "detail": str(exc)
        }), 500


# -----------------------------
# Controlled fault injection
# -----------------------------

@app.post("/fault")
def inject_fault():

    # Fault injection is disabled by default.
    if os.environ.get("ENABLE_FAULT_INJECTION") != "true":
        return jsonify({
            "error": "Fault injection disabled"
        }), 403

    # Require an authorization token.
    expected_token = os.environ.get("FAULT_TOKEN")
    provided_token = request.headers.get("X-Fault-Token")

    if not expected_token or provided_token != expected_token:
        return jsonify({
            "error": "Unauthorized"
        }), 401

    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify({
            "error": "Expected JSON object"
        }), 400

    fault_type = body.get("type")

    # Simulate an application crash.
    if fault_type == "crash":
        print(
            "FAULT INJECTION: Simulating application crash",
            flush=True
        )
        os._exit(1)

    return jsonify({
        "error": "Unsupported fault type"
    }), 400


# -----------------------------
# Application startup
# -----------------------------

load_model()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )
