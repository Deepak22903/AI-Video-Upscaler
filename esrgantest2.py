# esrgantest_private.py
from gradio_client import Client, handle_file
import os
import requests
import shutil

# Option A: read token from env
HF_TOKEN = os.environ.get("HF_TOKEN")  # export HF_TOKEN="hf_xxx" before running

# Option B: hardcode (not recommended)
# HF_TOKEN = "hf_xxx"

# Use the exact repo_id shown in the docs:
client = Client("deepak-6969/upscale_images", hf_token=HF_TOKEN)

# Call the endpoint (api_name should match the docs '/upscale_x2')
result = client.predict(
    input_img=handle_file(
        "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
    ),
    api_name="/upscale_x2",
)

# `result` is the returned tuple described in the docs.
# According to the docs:
#   result[0] -> dict with keys like 'path' and 'url'
#   result[1] -> filepath (for Download button)
print("raw result:", result)

# Save the returned file if present
if isinstance(result, (list, tuple)) and len(result) > 1 and result[1]:
    output_path = result[1]
    print(f"Upscaled image path: {output_path}")
    # copy the file to the current directory
    shutil.copy(output_path, "upscaled_image.png")
    print("Saved as upscaled_image.png")
else:
    print("No output path found in result. Inspect `result` above.")
