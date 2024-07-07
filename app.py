import io
import json
import os
import random
import time

import gradio as gr
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

load_dotenv()

COMFY_ADDRESS = os.getenv("COMFY_ADDRESS")
HISTORY_ENDPOINT = "history"
PROMPT_ENDPOINT = "prompt"
VIEW_ENDPOINT = "view"
UPLOAD_IMAGE_ENDPOINT = "upload/image"

BLACK_WORKFLOW = "black_workflow.json"
WHITE_WORKFLOW = "white_workflow.json"


def queue_prompt(prompt):
    server_address = COMFY_ADDRESS + PROMPT_ENDPOINT
    p = {"prompt": prompt}
    data = json.dumps(p).encode("utf-8")
    print(f"Sending prompt to {server_address}")
    print(f"Prompt data: {data}")
    try:
        response = requests.post(server_address, data=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"Queue prompt response: {result}")
        return result
    except Exception as e:
        print(f"Error in queue_prompt: {str(e)}")
        return None


def load_workflow(workflow_path):
    random_num = random.randrange(0, 1_000_000)
    with open(workflow_path, "r") as file:
        workflow = json.load(file)
        workflow["3"]["inputs"]["seed"] = random_num
        return workflow

# loading checkpoint takes a long time
def get_history(prompt_id, max_retries=20, delay=5):
    server_address = COMFY_ADDRESS + HISTORY_ENDPOINT
    server_address += "/" + prompt_id
    for attempt in range(max_retries):
        response = requests.get(server_address)

        if response.status_code == 200:
            history = response.json()
            if prompt_id in history:
                return history

        print(
            f"Attempt {attempt + 1}/{max_retries}: History not ready. Retrying in {delay} seconds..."
        )
        time.sleep(delay)

    raise Exception(
        f"History not available for prompt ID {prompt_id} after {max_retries} attempts."
    )


def get_image(filename, subfolder, folder_type):
    server_address = COMFY_ADDRESS + VIEW_ENDPOINT
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(server_address, params=params, timeout=10)
    return response.content


def get_images(prompt_id):
    output_images = []

    history = get_history(prompt_id)[prompt_id]
    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        output_data = {}
        if "images" in node_output:
            for image in node_output["images"]:
                if image["type"] == "output":
                    image_data = get_image(
                        image["filename"], image["subfolder"], image["type"]
                    )
                    output_data["image_data"] = image_data
        output_data["file_name"] = image["filename"]
        output_data["type"] = image["type"]
        output_images.append(output_data)

    return output_images


def upload_image(numpy_img: np.ndarray, mask: bool) -> dict | None:
    server_address = COMFY_ADDRESS + UPLOAD_IMAGE_ENDPOINT
    if mask:
        file_name = "mask.png"
    else:
        file_name = "image.png"

    image_bytes = numpy_to_bytes(numpy_img)
    multipart_data = MultipartEncoder(
        fields={
            "image": (file_name, image_bytes, "image/png"),
            "type": "input",
            "overwrite": "true",
        }
    )

    headers = {"Content-Type": multipart_data.content_type}

    response = requests.post(server_address, data=multipart_data, headers=headers, timeout=10)
    try:
        return response.json()
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        return None


def get_PIL_images(images):
    return [Image.open(io.BytesIO(image_data["image_data"])) for image_data in images]


def numpy_to_bytes(numpy_img: np.ndarray):
    img = Image.fromarray(numpy_img.astype("uint8"))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format="PNG")
    return byte_arr.getvalue()


def generate_image(numpy_img: np.ndarray, numpy_mask: np.ndarray, workflow: str):
    try:
        # Step 1: Upload the image and mask
        upload_img_result = upload_image(numpy_img, mask=False)
        print("Upload image result:", upload_img_result)

        upload_mask_result = upload_image(numpy_mask, mask=True)
        print("Upload mask result:", upload_mask_result)

        # Step 2: Load and modify the workflow
        workflow = load_workflow(workflow)
        image_path = (
            f"{upload_img_result['subfolder']}/{upload_img_result['name']}".strip("/")
        )
        mask_path = (
            f"{upload_mask_result['subfolder']}/{upload_mask_result['name']}".strip("/")
        )
        workflow["11"]["inputs"]["image"] = image_path
        workflow["54"]["inputs"]["image"] = mask_path

        # Step 3: Queue the prompt
        queue_result = queue_prompt(workflow)
        if not queue_result or "prompt_id" not in queue_result:
            print("Error: Failed to queue prompt. Result:", queue_result)
            return None

        prompt_id = queue_result["prompt_id"]
        print(f"Queued prompt with ID: {prompt_id}")

        # Step 4: Get the generated images
        images = get_images(prompt_id)
        if not images:
            print("Error: No images returned from get_images")
            return None
        print(f"Retrieved {len(images)} images")

        # Step 5: Convert to PIL images
        pil_images = get_PIL_images(images)
        if not pil_images:
            print("Error: No PIL images created")
            return None
        print(f"Created {len(pil_images)} PIL images")

        # Return the last image
        return pil_images[-1]

    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        return None

def check_mask(mask: np.ndarray) -> str:    
    # for some reason when the mask is black, num white pixels == num non-zero in alpha channel
    alpha_channel = mask[:,:,3]
    if np.count_nonzero(mask != 0) == np.count_nonzero(alpha_channel):
        return BLACK_WORKFLOW
    else:
        return WHITE_WORKFLOW    

def parse_gradio(canvas_dict) -> Image.Image | None:
    base_image = canvas_dict["background"]
    mask = canvas_dict["layers"][0]
    workflow = check_mask(mask)
    return generate_image(base_image, mask, workflow)


iface = gr.Interface(
    fn=parse_gradio,
    inputs=[
        gr.ImageEditor(
            type="numpy",
            sources=["upload", "clipboard"],
            brush=gr.Brush(color_mode="fixed", colors=["rgb(0,0,0)", "rgb(255,255,255)"]),
        )
    ],
    outputs=gr.Image(label="Output Image"),
    title="Tattoo Eraser",
    description="Upload an image, choose EITHER white or black brush, draw over tatoos, then get back image with tatoos removed.",
)

iface.launch()
