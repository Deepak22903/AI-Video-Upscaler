import os
import subprocess
import argparse
import cv2
from moviepy import VideoFileClip, ImageSequenceClip, AudioFileClip
import shutil
import json
from pathlib import Path
import httpx
import logging
import colorlog
from gradio_client import Client, handle_file
import requests
from pathlib import Path
import time
from httpx import RequestError, ConnectError
import threading

logger = colorlog.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REALESRGAN_DIR = os.path.join(BASE_DIR, "Real-ESRGAN")
REALESRGAN_SCRIPT = os.path.join(REALESRGAN_DIR, "inference_realesrgan.py")
REALESRGAN_MODEL = "RealESRGAN_x4plus"
REALESRGAN_MODEL_PATH = os.path.join(
    REALESRGAN_DIR, "experiments", "pretrained_models", f"{REALESRGAN_MODEL}.pth"
)

# Temporary directory setup
TEMP_DIR = os.path.join(BASE_DIR, "temp")
INPUT_FRAMES_DIR = os.path.join(TEMP_DIR, "input_frames")
OUTPUT_FRAMES_DIR = os.path.join(TEMP_DIR, "output_frames")
TEMP_AUDIO_FILE = os.path.join(TEMP_DIR, "original_audio.mp3")


def get_video_info(video_path):
    """Extract video metadata."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }
        cap.release()

        logger.info(
            f"Video info: {info['width']}x{info['height']} @ {info['fps']:.2f} FPS, "
            f"{info['frame_count']} frames, {info['duration']:.2f}s duration"
        )
        return info
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        raise


def extract_audio(input_path):
    """Extract audio from video file."""
    logger.info(f"Extracting audio from {input_path}...")
    try:
        video_clip = VideoFileClip(input_path)
        if video_clip.audio:
            video_clip.audio.write_audiofile(
                TEMP_AUDIO_FILE,
                codec="mp3",
                logger=None,  # Suppress moviepy's verbose logging
            )
            video_clip.close()
            logger.info("Audio extracted successfully.")
            return True
        else:
            logger.info("No audio track found in the video.")
            video_clip.close()
            return False
    except Exception as e:
        logger.warning(f"Could not extract audio: {e}")
        logger.info("Proceeding without audio.")
        return False


def extract_frames(input_path, video_info, max_frames=None):
    """Extract frames from video file."""
    logger.info(f"Extracting frames from {input_path}...")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_info["frame_count"])
    if max_frames:
        total_frames = min(total_frames, max_frames)

    frame_count = 0
    last_logged_percent = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_count >= max_frames:
            logger.info(f"Reached maximum frame limit: {max_frames}")
            break

        frame_filename = os.path.join(INPUT_FRAMES_DIR, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        # Log progress
        percent_done = (frame_count / total_frames) * 100
        if int(percent_done) // 10 > last_logged_percent:
            last_logged_percent = int(percent_done) // 10
            logger.info(f"Frame extraction: {int(percent_done)}% complete ({frame_count}/{total_frames} frames)")

    cap.release()
    logger.info(f"Extracted {frame_count} frames at {fps:.2f} FPS.")

    if frame_count == 0:
        raise ValueError("Video file is empty or corrupt.")

    return fps, frame_count


def run_enhancement(model_name, scale=4, face_enhance=True, enhancer="local"):
    """Run Real-ESRGAN enhancement on extracted frames."""
    logger.info(f"Running enhancement using {enhancer} method...")

    if enhancer == "api":
        # API-based enhancement
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

        client = Client("deepak-6969/upscale_images", hf_token=HF_TOKEN)

        input_frames = sorted([f for f in os.listdir(INPUT_FRAMES_DIR) if f.endswith(".png")])
        total_frames = len(input_frames)

        max_retries = 3
        retry_delay = 5  # seconds

        for i, frame_name in enumerate(input_frames):
            input_frame_path = os.path.join(INPUT_FRAMES_DIR, frame_name)
            logger.info(f"Processing frame {i + 1}/{total_frames} via API...")

            # Retry mechanism for API call
            for attempt in range(max_retries):
                try:
                    result = client.predict(
                        input_img=handle_file(input_frame_path),
                        api_name="/upscale_x2",
                    )
                    break  # Exit loop if successful
                except RequestError as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise

            if isinstance(result, (list, tuple)) and len(result) > 1 and result[1]:
                output_path = result[1]
                shutil.copy(output_path, os.path.join(OUTPUT_FRAMES_DIR, frame_name))
                logger.info(f"Saved enhanced frame {i + 1}/{total_frames}")
            else:
                logger.error(f"Failed to enhance {frame_name}. Result: {result}")
        logger.info("API enhancement complete.")
        return True

    elif enhancer == "local":
        # Local enhancement
        command = [
            "python",
            "-u",
            REALESRGAN_SCRIPT,
            "-n",
            model_name,
            "-i",
            INPUT_FRAMES_DIR,
            "-o",
            OUTPUT_FRAMES_DIR,
            "--outscale",
            str(scale),
        ]

        if face_enhance:
            command.append("--face_enhance")

        # Check if model file exists
        if not os.path.exists(REALESRGAN_MODEL_PATH):
            logger.warning(f"Model file not found at {REALESRGAN_MODEL_PATH}.")
            logger.warning("Please ensure you have downloaded the pre-trained models.")

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Attempt to parse progress from the output
                    if "Processing" in output and "..." in output:
                        try:
                            progress_part = output.split(" ")[1].split("/")
                            current = int(progress_part[0])
                            total = int(progress_part[1].split("]")[0])
                            percent = (current / total) * 100
                            logger.info(f"Local enhancement: {int(percent)}% complete ({current}/{total} frames)")
                        except (ValueError, IndexError):
                            logger.info(f"Real-ESRGAN: {output.strip()}")
                    else:
                        logger.info(f"Real-ESRGAN: {output.strip()}")
            rc = process.poll()

            if rc != 0:
                error_msg = process.stderr.read()
                raise subprocess.CalledProcessError(rc, command, stderr=error_msg)

            logger.info("Frame enhancement complete.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Enhancement failed: {e.stderr}")
            raise
    elif enhancer == "hybrid":
        # Hybrid enhancement: Split frames between local and API
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

        client = Client("deepak-6969/upscale_images", hf_token=HF_TOKEN)

        input_frames = sorted([f for f in os.listdir(INPUT_FRAMES_DIR) if f.endswith(".png")])
        total_frames = len(input_frames)

        # Split frames into two groups
        mid_index = total_frames // 2
        local_frames = input_frames[:mid_index]
        api_frames = input_frames[mid_index:]

        def process_local():
            logger.info("Starting local enhancement...")
            local_command = [
                "python",
                "-u",
                REALESRGAN_SCRIPT,
                "-n",
                model_name,
                "-i",
                INPUT_FRAMES_DIR,
                "-o",
                OUTPUT_FRAMES_DIR,
                "--outscale",
                str(scale),
                "--tile",
                "256",  # Use a smaller tile size to reduce memory usage
            ]

            if face_enhance:
                local_command.append("--face_enhance")

            try:
                process = subprocess.Popen(local_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                process.wait()
                if process.returncode != 0:
                    error_msg = process.stderr.read()
                    logger.error(f"Local enhancement failed: {error_msg}")
                    raise subprocess.CalledProcessError(process.returncode, local_command, stderr=error_msg)
            except Exception as e:
                logger.error(f"Error in local enhancement: {e}")

        def process_api():
            logger.info("Starting API enhancement...")
            for i, frame_name in enumerate(api_frames):
                input_frame_path = os.path.join(INPUT_FRAMES_DIR, frame_name)
                logger.info(f"Processing frame {i + 1}/{len(api_frames)} via API...")

                for attempt in range(3):
                    try:
                        result = client.predict(
                            input_img=handle_file(input_frame_path),
                            api_name="/upscale_x2",
                        )
                        if isinstance(result, (list, tuple)) and len(result) > 1 and result[1]:
                            output_path = result[1]
                            shutil.copy(output_path, os.path.join(OUTPUT_FRAMES_DIR, frame_name))
                            logger.info(f"Saved enhanced frame {i + 1}/{len(api_frames)}")
                        break
                    except ConnectError as e:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}")
                        if attempt < 2:
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            logger.error(f"API enhancement failed for frame {frame_name}: {e}")
                            return False
                    except Exception as e:
                        logger.error(f"Unexpected error during API enhancement: {e}")
                        return False
            return True

        # Create threads for local and API processing
        local_thread = threading.Thread(target=process_local)
        api_thread = threading.Thread(target=process_api)

        # Start threads
        local_thread.start()
        api_thread.start()

        # Wait for both threads to complete
        local_thread.join()
        api_thread.join()

        logger.info("Hybrid enhancement complete.")
        return True


def get_enhanced_frames():
    """Get list of enhanced frame files."""
    # Try standard output first
    frames = [
        os.path.join(OUTPUT_FRAMES_DIR, f)
        for f in sorted(os.listdir(OUTPUT_FRAMES_DIR))
        if f.endswith(".png")
    ]

    # If none found, try _out suffix
    if not frames:
        frames = [
            os.path.join(OUTPUT_FRAMES_DIR, f)
            for f in sorted(os.listdir(OUTPUT_FRAMES_DIR))
            if f.startswith("frame_") and f.endswith("_out.png")
        ]

    if not frames:
        raise RuntimeError("No enhanced frames found in output directory.")

    logger.info(f"Found {len(frames)} enhanced frames.")
    return frames


def reassemble_video(enhanced_frames, fps, output_path, has_audio=False):
    """Reassemble enhanced frames into final video."""
    logger.info("Reassembling enhanced video...")

    # Create video clip from image sequence - USE ALL FRAMES
    final_clip = ImageSequenceClip(enhanced_frames, fps=fps)

    # Add audio if available
    if has_audio and os.path.exists(TEMP_AUDIO_FILE):
        logger.info("Adding original audio...")
        original_audio = AudioFileClip(TEMP_AUDIO_FILE)
        # Ensure audio duration matches video duration
        if original_audio.duration > final_clip.duration:
            original_audio = original_audio.subclipped(0, final_clip.duration)
        final_clip = final_clip.with_audio(original_audio)

    # Write final video
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac" if has_audio else None,
        logger="bar",  # Use moviepy's progress bar
        preset="medium",
        bitrate="5000k",
    )

    final_clip.close()
    if has_audio and os.path.exists(TEMP_AUDIO_FILE):
        original_audio.close()

    logger.info(f"Successfully saved enhanced video to {output_path}")


# Also fix the extract_frames call in enhance_video function:
def enhance_video(input_path, output_path, max_frames=None, scale=4, face_enhance=True, enhancer="local"):
    """
    Main function to enhance a video file.

    Args:
        input_path: Path to input video
        output_path: Path to save enhanced video
        max_frames: Maximum number of frames to process (None for all)
        scale: Upscaling factor (default: 4)
        face_enhance: Enable face enhancement (default: True)
        enhancer: Enhancement method to use (local or api)
    """
    try:
        # Setup: Clean and create temp directories
        logger.info("Setting up temporary directories...")
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(INPUT_FRAMES_DIR, exist_ok=True)
        os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

        # Get video information
        video_info = get_video_info(input_path)

        # Extract audio
        has_audio = extract_audio(input_path)

        # Extract frames - FIX: Use the max_frames parameter
        fps, frame_count = extract_frames(input_path, video_info, max_frames=max_frames)

        # Run enhancement
        run_enhancement(REALESRGAN_MODEL, scale, face_enhance, enhancer)

        # Get enhanced frames
        enhanced_frames = get_enhanced_frames()

        # Reassemble video
        reassemble_video(enhanced_frames, fps, output_path, has_audio)

        logger.info("Video enhancement completed successfully!")
        return True

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
    finally:
        cleanup_temp_files()


def cleanup_temp_files():
    """Clean up temporary directories."""
    logger.info("Cleaning up temporary files...")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    logger.info("Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhance a video using Real-ESRGAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to save the enhanced output video",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)",
    )
    parser.add_argument(
        "--scale", type=int, default=4, choices=[2, 4], help="Upscaling factor"
    )
    parser.add_argument(
        "--no-face-enhance", action="store_true", help="Disable face enhancement"
    )
    parser.add_argument(
        "--enhancer",
        type=str,
        default="local",
        choices=["local", "api", "hybrid"],
        help="Enhancement method to use (local, api, or hybrid)",
    )
    parser.add_argument(
        "--simple-logging", action="store_true", help="Enable simple logging for use as a module"
    )

    args = parser.parse_args()

    # Configure logging
    if args.simple_logging:
        # Basic configuration for when imported as a module
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
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
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        exit(1)

    # Validate Real-ESRGAN script
    if args.enhancer == "local" and not os.path.exists(REALESRGAN_SCRIPT):
        logger.error(f"Real-ESRGAN script not found: {REALESRGAN_SCRIPT}")
        logger.error("Please clone the Real-ESRGAN repo into the same directory.")
        exit(1)

    # Run enhancement
    try:
        enhance_video(
            args.input,
            args.output,
            max_frames=args.max_frames,
            scale=args.scale,
            face_enhance=not args.no_face_enhance,
            enhancer=args.enhancer,
        )
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        exit(1)

