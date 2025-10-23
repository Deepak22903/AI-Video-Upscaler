# Video Enhancer Web App

A web application for upscaling and enhancing videos using Real-ESRGAN.

## Features

- **Video Upscaling**: Enhance video resolution with state-of-the-art AI models.
- **Face Enhancement**: Improve facial details in videos.
- **Asynchronous Processing**: Upload a video and track the enhancement progress via a job ID.
- **RESTful API**: Easily integrate the enhancement service into your own applications.
- **Simple Web Interface**: A straightforward UI for uploading and managing videos.

## Technologies Used

- **Backend**: Python, Flask
- **AI Model**: Real-ESRGAN
- **Video Processing**: moviepy, OpenCV
- **Dependencies**: PyTorch, basicsr, gfpgan, and more.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Deepak22903/AI-Video-Upscaler
    cd https://github.com/Deepak22903/AI-Video-Upscaler
    ```

2.  **Create a virtual environment and activate it:**

    ```bash
    python3.11 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Pre-trained Models:**

    This project requires pre-trained models for Real-ESRGAN and GFPGAN.

    - **Real-ESRGAN Model**:
      The `RealESRGAN_x4plus.pth` model should be present in the `Real-ESRGAN/weights/` directory.

    - **GFPGAN Models**:
      The `detection_Resnet50_Final.pth` and `parsing_parsenet.pth` models should be present in the `gfpgan/weights/` directory. These are used for face enhancement.

## Usage

1.  **Start the Flask server:**

    ```bash
    python3.11 app.py
    ```

2.  **Access the application:**

    Open your web browser and navigate to `http://127.0.0.1:5000`.

## API Endpoints

- `POST /enhance`

  - Upload a video for enhancement.
  - **Form Data**: `file` (the video file)
  - **Returns**: A JSON object with the `job_id`.

- `GET /status/<job_id>`

  - Check the status of an enhancement job.
  - **Returns**: A JSON object with the job status (`queued`, `processing`, `complete`, `failed`), progress, and a `download_url` when complete.

- `GET /download/<filename>`
  - Download the enhanced video.
