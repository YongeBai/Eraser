import json
import os

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

COMFY_API = os.getenv("COMFY_API")
OUTPUT_DIR = "~/ComfyUI/output"
WORKFLOW = "workflow_api.json"

def queue_prompt(workflow: str):
    prompt = {"prompt": workflow}
    data = json.dumps(prompt).encode("utf-8")
    requests.post(COMFY_API, data=data, timeout=10)

def generate():
    with open(WORKFLOW, "r") as f:
        workflow = json.load(f)

    queue_prompt(workflow)

demo = gr.Interface(fn=generate, title="ComfyUI", description="Generate a new UI workflow", inputs=[], outputs=["image"])

demo.launch()
