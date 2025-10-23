import replicate

input = {
    "prompt": "Make the sheets in the style of the logo. Make the scene natural. ",
    "image_input": [
        "https://replicate.delivery/pbxt/NbYIclp4A5HWLsJ8lF5KgiYSNaLBBT1jUcYcHYQmN1uy5OnN/tmpcqc07f_q.png",
        "https://replicate.delivery/pbxt/NbYId45yH8s04sptdtPcGqFIhV7zS5GTcdS3TtNliyTAoYPO/Screenshot%202025-08-26%20at%205.30.12%E2%80%AFPM.png",
    ],
}

output = replicate.run("google/nano-banana", input=input)

# To access the file URL:
print(output.url())
# => "https://replicate.delivery/.../output.jpg"

# To write the file to disk:
with open("output.jpg", "wb") as file:
    file.write(output.read())
# => output.jpg written to disk
