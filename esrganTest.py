from gradio_client import Client, handle_file
import shutil

client = Client("salman555/upscale_images")
result = client.predict(
    input_img=handle_file(
        "input_rgb_safe.png"
    ),
    api_name="/upscale_x2",
)
shutil.copy(result[1], "upscaled_image.png")
