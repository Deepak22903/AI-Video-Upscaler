from gradio_client import Client, handle_file

client = Client("Nick088/Real-ESRGAN_Pytorch")
result = client.predict(
    img=handle_file(
        "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
    ),
    size_modifier="4",
    api_name="/predict",
)
print(result)
