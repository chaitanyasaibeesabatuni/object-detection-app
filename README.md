# AI Object Detection Microservice

This project is a simple AI-powered object detection microservice. It allows you to upload images, detect objects in them, and get the results with bounding boxes and confidence scores. The service is built using Python and Docker for easy deployment.

---

## Features

* Detect multiple objects in images
* Returns object type, confidence score, and bounding box coordinates
* Supports batch image uploads
* Easy to run locally or in a containerized environment

---

## Project Structure

* **ai-backend**: Contains the main backend code for object detection (`app.py`)
* **ui-backend**: Contains the web interface to upload images and view results (`app.py`)
* **uploads**: Folder where you can place images to be processed
* **outputs**: Folder where processed images and detection results are saved
* **docker-compose.yml**: File to run the service using Docker

---

## Requirements

* Docker and Docker Compose installed
* Python 3.10 or higher if running locally
* Required Python libraries listed in `requirements.txt` for both backends

---

## Getting Started

### Using Docker Compose

1. Clone the repository.
2. Run the project using Docker Compose:

```bash
docker-compose up --build
```

3. Open your browser and go to:

```
http://localhost:5001
```

4. Place images in the `uploads` folder. Processed results will appear in the `outputs` folder.

### Running Without Docker

1. Install dependencies for AI backend:

```bash
cd ai-backend
pip install -r requirements.txt
```

2. Start the AI backend:

```bash
python app.py
```

3. Install dependencies for UI backend:

```bash
cd ../ui-backend
pip install -r requirements.txt
```

4. Start the UI backend:

```bash
python app.py
```

5. Open your browser and go to `http://localhost:5001`
6. Place images in the `uploads` folder and check the processed images in the `outputs` folder.

> **Note:** Both backends must be running for the project to work properly.

---

## How to Use

* Upload images via the web interface or API
* The service will detect objects and return results in JSON format
* Each detected object includes:

  * Object name
  * Confidence score
  * Bounding box coordinates

> Output images with detected objects are stored in the `outputs` folder. You can open any processed image to see the detection in action.

---

## Contributing

If you want to contribute:

* Fork the repository
* Make your changes
* Create a pull request

---

## License

This project is open source and free to use for personal or commercial purposes.
