# esrgan_test_remote.py
# pip install gradio_client
from gradio_client import Client, handle_file
import traceback

client = Client("Nick088/Real-ESRGAN_Pytorch")

# Known example image used in many docs — replace with any stable URL
local_image = "/mnt/deepak/data/btechProject/Real-ESRGAN/inputs/0014.jpg"

try:
    print("Calling /predict with local image...")
    result = client.predict(
        img=handle_file(local_image), size_modifier="4", api_name="/predict"
    )
    print("Raw result:", type(result), result)
except Exception as e:
    print("Prediction failed:", type(e).__name__, e)
    traceback.print_exc()
