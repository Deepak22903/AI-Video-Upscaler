import os
import threading
import uuid
import subprocess
import time
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Dummy comment to force reload
# --- Configuration ---
from werkzeug.utils import secure_filename
import logging
import colorlog

# Configure logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Configuration
UPLOAD_FOLDER = "uploads"
ENHANCED_FOLDER = "enhanced"
TEMP_FOLDER = "temp"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi"}
MAX_FRAMES_DEFAULT = 300  # Limit frame processing for reasonable processing time

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ENHANCED_FOLDER"] = ENHANCED_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Job tracking
jobs = {}
jobs_lock = threading.Lock()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def update_job_status(job_id, status, progress=None, error=None):
    """Update job status in thread-safe manner."""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = status
            jobs[job_id]["updated_at"] = datetime.now().isoformat()
            if progress is not None:
                jobs[job_id]["progress"] = progress
            if error is not None:
                jobs[job_id]["error"] = error


def enhance_video_task(job_id, input_path, output_path, max_frames=None):
    """
    Background task to run the video enhancement.
    """
    logger.info(f"Starting enhancement task for job: {job_id}")
    update_job_status(job_id, "processing", progress=0)

    try:
        # Build command
        command = [
            "python",
            "video_enhancer.py",
            "--input",
            input_path,
            "--output",
            output_path,
        ]

        if max_frames:
            command.extend(["--max-frames", str(max_frames)])

        # Run enhancement
        update_job_status(job_id, "processing", progress=10)

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Monitor process
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            logger.info(f"Enhancement completed for job: {job_id}")
            update_job_status(job_id, "complete", progress=100)
        else:
            error_msg = stderr or "Enhancement process failed"
            logger.error(f"Enhancement failed for job {job_id}: {error_msg}")
            update_job_status(job_id, "failed", error=error_msg)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in enhancement task for job {job_id}: {error_msg}")
        update_job_status(job_id, "failed", error=error_msg)


@app.route("/enhance", methods=["POST"])
def enhance_video_endpoint():
    """
    API endpoint to upload a video and start enhancement.
    """
    # Check if file is present
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    # Check if filename is empty
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    if not allowed_file(file.filename):
        return jsonify(
            {"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}
        ), 400

    # Get optional parameters
    max_frames = request.form.get("max_frames", MAX_FRAMES_DEFAULT, type=int)

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    filename = f"{job_id}{ext}"

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    output_path = os.path.join(app.config["ENHANCED_FOLDER"], filename)

    # Save uploaded file
    try:
        file.save(input_path)
        file_size = os.path.getsize(input_path)
        logger.info(f"File saved: {filename} ({file_size} bytes)")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    # Create job entry
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": filename,
            "status": "queued",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "input_path": input_path,
            "output_path": output_path,
            "error": None,
        }

    # Start background thread
    thread = threading.Thread(
        target=enhance_video_task,
        args=(job_id, input_path, output_path, max_frames),
        daemon=True,
    )
    thread.start()

    logger.info(f"Started enhancement job: {job_id}")
    return jsonify({"job_id": job_id, "filename": filename, "status": "queued"}), 202


@app.route("/status/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """
    Get status of a specific job.
    """
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404

        job = jobs[job_id].copy()

    # Build response
    response = {
        "job_id": job["id"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }

    if job["status"] == "complete":
        response["download_url"] = f"/download/{job['filename']}"
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")

    return jsonify(response)


@app.route("/download/<filename>", methods=["GET"])
def download_enhanced_video(filename):
    """
    Download enhanced video file.
    """
    file_path = os.path.join(app.config["ENHANCED_FOLDER"], filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(
        app.config["ENHANCED_FOLDER"],
        filename,
        as_attachment=True,
        download_name=f"enhanced_{filename}",
    )


@app.route("/preview/<filename>", methods=["GET"])
def preview_video(filename):
    """
    Stream video for preview.
    """
    file_path = os.path.join(app.config["ENHANCED_FOLDER"], filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(app.config["ENHANCED_FOLDER"], filename)


@app.route("/jobs", methods=["GET"])
def list_jobs():
    """
    List all jobs (for debugging/admin).
    """
    with jobs_lock:
        job_list = [
            {
                "id": job["id"],
                "status": job["status"],
                "progress": job.get("progress", 0),
                "created_at": job["created_at"],
            }
            for job in jobs.values()
        ]

    return jsonify({"jobs": job_list})


@app.route("/cleanup/<job_id>", methods=["DELETE"])
def cleanup_job(job_id):
    """
    Clean up job files.
    """
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404

        job = jobs[job_id]
        input_path = job["input_path"]
        output_path = job["output_path"]

    # Delete files
    try:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

        with jobs_lock:
            del jobs[job_id]

        logger.info(f"Cleaned up job: {job_id}")
        return jsonify({"message": "Job cleaned up successfully"})
    except Exception as e:
        logger.error(f"Error cleaning up job {job_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """Serve the main HTML page."""
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify(
        {"error": f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"}
    ), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting Video Enhancement Server...")
    logger.info(f"Max file size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB")
    logger.info(f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")

    # Run server
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
