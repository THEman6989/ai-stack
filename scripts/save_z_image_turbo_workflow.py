#!/usr/bin/env python3
"""Save the Z-Image Turbo workflow into the AlphaRavis ComfyUI workflow library.

Run this inside the langgraph-api container after restart:

    docker compose exec langgraph-api python /workspace/scripts/save_z_image_turbo_workflow.py
"""

import json
import sys
from pathlib import Path

for candidate in ("/app", "/workspace/langgraph-app"):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
from comfyui_workflow_library import save_comfyui_workflow_record, list_comfyui_workflow_records


WORKFLOW_JSON = r"""{
  "60": {
    "inputs": { "filename_prefix": "z-image-turbo", "images": ["83:8", 0] },
    "class_type": "SaveImage", "_meta": { "title": "Save Image" }
  },
  "83:30": {
    "inputs": { "clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default" },
    "class_type": "CLIPLoader", "_meta": { "title": "Load CLIP" }
  },
  "83:13": {
    "inputs": { "width": 1024, "height": 1024, "batch_size": 1 },
    "class_type": "EmptySD3LatentImage", "_meta": { "title": "EmptySD3LatentImage" }
  },
  "83:33": {
    "inputs": { "conditioning": ["83:27", 0] },
    "class_type": "ConditioningZeroOut", "_meta": { "title": "ConditioningZeroOut" }
  },
  "83:8": {
    "inputs": { "samples": ["83:3", 0], "vae": ["83:29", 0] },
    "class_type": "VAEDecode", "_meta": { "title": "VAE Decode" }
  },
  "83:3": {
    "inputs": {
      "seed": 465454499048436, "steps": 4, "cfg": 1,
      "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1,
      "model": ["83:28", 0], "positive": ["83:27", 0],
      "negative": ["83:33", 0], "latent_image": ["83:13", 0]
    },
    "class_type": "KSampler", "_meta": { "title": "KSampler" }
  },
  "83:27": {
    "inputs": {
      "text": "Giant blue and purple big billboard on rooftop in san francisco city billboard says \"ComfyUI is built with love\" All kinds of buoildings in different shapes and colors. Some buildings have grafitti \"We\" \"Here\" \"Today\"",
      "clip": ["83:30", 0]
    },
    "class_type": "CLIPTextEncode", "_meta": { "title": "CLIP Text Encode (Prompt)" }
  },
  "83:28": {
    "inputs": { "unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default" },
    "class_type": "UNETLoader", "_meta": { "title": "Load Diffusion Model" }
  },
  "83:29": {
    "inputs": { "vae_name": "z-image-ae.safetensors" },
    "class_type": "VAELoader", "_meta": { "title": "Load VAE" }
  }
}"""


def main():
    wf = json.loads(WORKFLOW_JSON)

    print("=== SAVE z_image_turbo ===")
    result = save_comfyui_workflow_record(
        workflow_name="z_image_turbo",
        workflow=wf,
        description="Z-Image Turbo text-to-image (SD3-style sampler, 1024x1024, 4 steps)",
        aliases=["z-image-turbo", "z image turbo", "zimage"],
        parameter_map={
            "prompt": "83:27.inputs.text",
            "seed": "83:3.inputs.seed",
            "steps": "83:3.inputs.steps",
            "cfg": "83:3.inputs.cfg",
            "width": "83:13.inputs.width",
            "height": "83:13.inputs.height",
        },
        workflow_type="image",
        source="ComfyUI getting-started example",
        overwrite=True,
    )
    print(json.dumps(result, indent=2, default=str))

    if result.get("ok"):
        print("\n=== VERIFY ===")
        lst = list_comfyui_workflow_records()
        print(f"{lst.get('count', 0)} workflow(s) saved.")
    else:
        print(f"\nERROR: {result.get('error', result.get('message', 'unknown'))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
