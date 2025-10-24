#!/usr/bin/env python3
from dotenv import load_dotenv
import os

print("Testing token loading...")
print(f"Shell HF_TOKEN: {os.environ.get('HF_TOKEN', 'Not set')[:20]}..." if os.environ.get('HF_TOKEN') else "Shell HF_TOKEN: Not set")

# Load from .env with override
load_dotenv(override=True)

token = os.getenv('HF_TOKEN')
print(f".env HF_TOKEN: {token[:20]}..." if token else ".env HF_TOKEN: Not found")

# Now test the Gradio client connection
try:
    from gradio_client import Client
    print(f"\nTrying to connect to deepak-6969/upscale_images with token...")
    client = Client("deepak-6969/upscale_images", hf_token=token)
    print("✓ Connection successful!")
except Exception as e:
    print(f"✗ Connection failed: {type(e).__name__}: {str(e)[:100]}")
