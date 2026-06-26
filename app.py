import os
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tempfile
import traceback
import mediapipe as mp
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "skintone_model.onnx"
cls_names  = ['dark', 'fair', 'light']
INPUT_NAME = "keras_tensor_352"

COLOR_MAP = {
    "white": "#FFFFFF", "cream": "#FFFDD0", "yellow": "#FFD700",
    "cyan": "#00CED1", "mint": "#98FF98", "sky blue": "#87CEEB",
    "bright red": "#FF2400", "lavender": "#E6E6FA", "pastel pink": "#FFD1DC",
    "light gray": "#D3D3D3", "maroon": "#800000", "teal": "#008080",
    "olive": "#808000", "charcoal": "#36454F", "mustard": "#FFDB58",
    "denim blue": "#1560BD", "forest green": "#228B22", "deep purple": "#673AB7",
    "navy": "#001F5B", "black": "#1a1a1a", "emerald": "#50C878",
    "burgundy": "#800020", "royal blue": "#4169E1", "deep green": "#006400",
    "rust": "#B7410E", "gold": "#FFD700", "brown": "#8B4513",
    "beige": "#F5F5DC", "coral": "#FF7F50", "silver": "#C0C0C0",
    "purple": "#800080", "gray": "#808080", "taupe": "#483C32",
}

L_CHK = [234, 93, 132, 58, 172]
R_CHK = [454, 323, 361, 288, 397]
FORH  = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397]

POOLS = {
    "dark": ["white","cream","yellow","cyan","mint","sky blue","bright red","lavender","pastel pink","light gray"],
    "mid":  ["maroon","teal","olive","charcoal","mustard","denim blue","forest green","deep purple"],
    "fair": ["navy","black","emerald","burgundy","royal blue","charcoal","maroon","teal","deep green"],
}
UNDERTONE_BONUS = {
    "warm":    ["olive","mustard","rust","gold","brown","beige","coral"],
    "cool":    ["navy","silver","emerald","purple","teal","charcoal"],
    "neutral": ["black","white","gray","denim blue","cream","taupe"],
}

# ── Load model once at startup ────────────────────────────────────────────────
sess = ort.InferenceSession(MODEL_PATH)

# ── Core functions ────────────────────────────────────────────────────────────
def pred_tone(path):
    img   = np.array(Image.open(path).convert("RGB").resize((224, 224))).astype(np.float32)
    probs = sess.run(None, {INPUT_NAME: np.expand_dims(img, 0)})[0][0]
    return cls_names[int(np.argmax(probs))], probs

def tone_value(probs):
    weights = {"dark": 0.0, "light": 0.5, "fair": 1.0}
    return sum(float(probs[i]) * weights[cls_names[i]] for i in range(len(cls_names)))

def extract_lab_and_overlay(img_path):
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None, None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
            res = fm.process(img_rgb)
            if not res.multi_face_landmarks:
                return None, None
            lms = res.multi_face_landmarks[0].landmark
            def roi_mask(ids):
                pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in ids], dtype=np.int32)
                m   = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(m, [pts], 255)
                return m
            mask     = roi_mask(L_CHK) | roi_mask(R_CHK) | roi_mask(FORH)
            skin_rgb = img_rgb[mask == 255]
            if len(skin_rgb) < 500:
                return None, None
            skin_lab = cv2.cvtColor(skin_rgb.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1,3)
            overlay  = img_rgb.copy()
            overlay[mask == 255] = (overlay[mask == 255] * 0.5 + np.array([255,160,80], dtype=np.float32) * 0.5).astype(np.uint8)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode('.jpg', overlay_bgr)
            overlay_b64 = base64.b64encode(buf).decode('utf-8')
            return skin_lab.mean(axis=0), overlay_b64
    except Exception:
        return None, None

def get_undertone(lab_mean):
    _, A, B = lab_mean
    if B - A >= 8: return "warm"
    if A - B >= 8: return "cool"
    return "neutral"

def recommend(tv, undertone=None, top_k=5):
    wd = max(0.0, 1.0 - 2.0 * tv)
    wm = 1.0 - abs(2.0 * tv - 1.0)
    wf = max(0.0, 2.0 * tv - 1.0)
    scores = {}
    for c in POOLS["dark"]: scores[c] = scores.get(c, 0) + wd
    for c in POOLS["mid"]:  scores[c] = scores.get(c, 0) + wm
    for c in POOLS["fair"]: scores[c] = scores.get(c, 0) + wf
    for c in UNDERTONE_BONUS.get(undertone, []):
        scores[c] = scores.get(c, 0) + 0.20
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        tone, probs  = pred_tone(tmp_path)
        tv           = tone_value(probs)
        lab, overlay = extract_lab_and_overlay(tmp_path)
        undertone    = get_undertone(lab) if lab is not None else "neutral"
        ranked       = recommend(tv, undertone)

        os.unlink(tmp_path)

        return jsonify({
            "tone":       tone,
            "tone_value": round(tv, 3),
            "probs":      {cls_names[i]: round(float(probs[i]) * 100, 1) for i in range(len(cls_names))},
            "undertone":  undertone,
            "colors":     [{"name": c, "hex": COLOR_MAP.get(c, "#ccc"), "score": round(s, 2)} for c, s in ranked],
            "overlay":    overlay,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)