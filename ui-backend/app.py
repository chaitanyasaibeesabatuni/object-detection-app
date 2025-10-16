from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import requests
from werkzeug.utils import secure_filename

# =========================================================
# Flask setup
# =========================================================
app = Flask(__name__)

# =========================================================
# Directory setup (relative to project root)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                 # ui_backend/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))          # project root
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Configure Flask paths
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["OUTPUT_FOLDER"] = OUTPUT_DIR

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# AI backend endpoint
AI_BACKEND_URL = "http://localhost:5001/detect"


# =========================================================
# Routes
# =========================================================
@app.route("/")
def index():
    """Render the main UI page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and forward it to AI backend for detection."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    print(f"File saved: {filepath}")

    try:
        with open(filepath, "rb") as f:
            response = requests.post(AI_BACKEND_URL, files={"file": f})
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with AI backend: {e}")
        return jsonify({"error": "Failed to reach AI backend"}), 500

    if response.status_code != 200:
        print(f"AI backend error: {response.text}")
        return jsonify({"error": "AI backend processing failed"}), 500

    data = response.json()
    print(f"Detection completed. Objects detected: {len(data.get('detections', []))}")
    return jsonify(data)


@app.route("/outputs/<filename>")
def get_output_image(filename):
    """Serve processed images from the outputs directory."""
    try:
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return jsonify({"error": "File not found"}), 404


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Starting UI Backend at http://localhost:5000")
    print(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
    print(f"Output directory: {app.config['OUTPUT_FOLDER']}")
    app.run(host="0.0.0.0", port=5000, debug=True)
