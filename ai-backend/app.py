import os
import cv2
import json
import numpy as np
import shutil
import torch
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# =========================================================
# Flask setup
# =========================================================
app = Flask(__name__)
CORS(app)

# =========================================================
# Directory setup (relative to project root)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # ai_backend/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # project root
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Uploads directory: {UPLOAD_DIR}")
print(f"Outputs directory: {OUTPUT_DIR}")

# =========================================================
# Model loading
# =========================================================
model = None
torch.backends.mkldnn.enabled = False
torch.backends.openmp.enabled = True


def load_model():
    """Load YOLOv8n model to CPU."""
    global model
    try:
        print("Loading YOLOv8n model on CPU...")
        model = YOLO("yolov8n.pt")
        model.to("cpu")
        print("Model loaded successfully.")

        # Quick self-test
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict(source=test_image, conf=0.1)
        print("Model self-test passed.")
        return True
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False


# =========================================================
# Detection logic
# =========================================================
def perform_detection(image_path):
    """Run YOLO detection on the given image."""
    try:
        print(f"Running detection on: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unable to read image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(source=img_rgb, conf=0.25, imgsz=640, device="cpu", verbose=False)
        result = results[0]
        boxes = result.boxes

        print(f"Detected {len(boxes)} objects")

        detections = []
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().flatten().tolist()
            conf = float(box.conf.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])
            name = result.names[cls]

            detections.append({
                "class": name,
                "confidence": round(conf, 3),
                "bbox": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2)
                }
            })
        return detections
    except Exception as e:
        traceback.print_exc()
        print(f"Detection error: {e}")
        return []


# =========================================================
# Draw bounding boxes
# =========================================================
def draw_bounding_boxes(image_path, detections, output_path):
    """Draw bounding boxes on image and save output."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not read image")
            return False

        for det in detections:
            bbox = det["bbox"]
            label = f"{det['class']} {det['confidence']:.2f}"
            x1, y1, x2, y2 = map(int, [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imwrite(output_path, img)
        print(f"Saved annotated image: {output_path}")
        return True
    except Exception as e:
        print(f"Error while drawing boxes: {e}")
        shutil.copy2(image_path, output_path)
        return True


# =========================================================
# Routes
# =========================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "AI backend running",
        "model_loaded": model is not None
    })


@app.route("/detect", methods=["POST"])
def detect():
    """Handle image upload and run detection."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        image_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(image_path)
        print(f"Uploaded file saved at: {image_path}")

        detections = perform_detection(image_path)
        base_name = os.path.splitext(file.filename)[0]
        output_image = os.path.join(OUTPUT_DIR, f"{base_name}_detected.jpg")
        output_json = os.path.join(OUTPUT_DIR, f"{base_name}_results.json")

        draw_bounding_boxes(image_path, detections, output_image)

        with open(output_json, "w") as f:
            json.dump(detections, f, indent=2)

        return jsonify({
            "success": True,
            "output_image": os.path.basename(output_image),
            "output_json": os.path.basename(output_json),
            "detections": detections
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Starting YOLOv8 Detection Backend...")
    if not load_model():
        print("Warning: Model could not be loaded.")
    print("Server running at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
