"""Gradio demo: find the faces in a photo and classify each face's expression.

Run:   python app.py      then open http://127.0.0.1:7860

Uses the static INT8 ONNX model from 05_optimization.ipynb (the fastest variant,
with 0.1 points less test accuracy than FP32) and the same preprocessing as training,
imported from inference.py.
"""
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort

from inference import EMOTIONS, preprocess

MODEL_PATH = Path(__file__).resolve().parent / "weights" / "resnet18_fer_int8.onnx"
FACE_DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
MARGIN = 0.1        # widen each face box by 10% per side, so the crop includes forehead and chin like FER2013
MIN_FACE = 48       # ignore detections smaller than the model's training images
BOX_COLOR = (0, 200, 255)


def load_session(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run 05_optimization.ipynb, or download resnet18_fer_int8.onnx "
            "from the GitHub Release into the weights/ folder (see README)."
        )
    options = ort.SessionOptions()
    options.add_session_config_entry("session.set_denormal_as_zero", "1")  # see notebook 05, section 2b
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


SESSION = load_session()
INPUT_NAME = SESSION.get_inputs()[0].name


def classify(face):
    """Grayscale face crop -> {emotion: probability}."""
    logits = SESSION.run(None, {INPUT_NAME: preprocess(face).numpy()})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return {emotion: float(p) for emotion, p in zip(EMOTIONS, probs)}


classify(np.zeros((48, 48), dtype=np.uint8))  # warm-up run, so the first real request isn't slower


def detect_faces(gray):
    """Face boxes (x1, y1, x2, y2), each widened by MARGIN and clipped to the image."""
    found = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE, MIN_FACE))
    height, width = gray.shape
    boxes = []
    for x, y, w, h in found:
        dx, dy = int(w * MARGIN), int(h * MARGIN)
        boxes.append((max(0, x - dx), max(0, y - dy), min(width, x + w + dx), min(height, y + h + dy)))
    return boxes


def draw_label(image, box, text, thickness):
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, thickness)
    scale = max(0.5, (x2 - x1) / 250)
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top = max(0, y1 - text_h - baseline - 4)
    cv2.rectangle(image, (x1, top), (x1 + text_w + 4, top + text_h + baseline + 4), BOX_COLOR, -1)
    cv2.putText(image, text, (x1 + 2, top + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)


def analyze(image):
    """RGB photo -> (photo with labeled boxes, probabilities for the largest face, info text)."""
    if image is None:
        return None, None, "Upload a photo or take one with the webcam."

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    boxes = detect_faces(gray)
    if not boxes:
        return image, None, "No face found. Try a front-facing photo with good lighting."

    annotated = image.copy()
    thickness = max(2, round(min(image.shape[:2]) / 300))
    results, model_ms = [], []
    for box in boxes:
        x1, y1, x2, y2 = box
        start = time.perf_counter()
        probs = classify(gray[y1:y2, x1:x2])
        model_ms.append((time.perf_counter() - start) * 1000)
        top = max(probs, key=probs.get)
        draw_label(annotated, box, f"{top} {probs[top]:.0%}", thickness)
        results.append(((x2 - x1) * (y2 - y1), probs))

    largest = max(results, key=lambda r: r[0])[1]
    info = (f"Found {len(boxes)} face(s). Probabilities are for the largest one. "
            f"Model time: {np.mean(model_ms):.1f} ms per face (INT8, CPU).")
    return annotated, largest, info


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="numpy", image_mode="RGB", sources=["upload", "webcam", "clipboard"], label="Photo"),
    outputs=[
        gr.Image(type="numpy", label="Detected faces"),
        gr.Label(num_top_classes=len(EMOTIONS), label="Largest face"),
        gr.Textbox(label="Info"),
    ],
    title="Facial Expression Recognition",
    description=(
        "Finds faces with OpenCV, then classifies 7 expressions with a ResNet18 fine-tuned on FER2013 "
        "and quantized to INT8 with ONNX Runtime. Works best on front-facing faces."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()