#!/usr/bin/env python3
"""
Quick end-to-end test for the LangSAM MQTT server.

Usage:
    pip install paho-mqtt numpy opencv-python
    MQTT_HOST=10.205.10.16 MQTT_USERNAME=dev MQTT_PASSWORD=<pass> \
        python test_segmentation.py /path/to/photo.jpg "water bottle."

The script sends the image to the server, waits for the mask, prints
pixel stats, and saves a green-overlay debug image to ./langsam_debug/.
"""

import os
import sys

import cv2

from lucid_langsam_client import LangSamClient

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: test_segmentation.py <image_path> [prompt]")

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "object."

    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot read image: {image_path}")

    host = os.environ.get("MQTT_HOST")
    if not host:
        sys.exit("Set MQTT_HOST, MQTT_USERNAME, MQTT_PASSWORD environment variables.")

    seg = LangSamClient(
        host=host,
        port=int(os.environ.get("MQTT_PORT", "1883")),
        username=os.environ.get("MQTT_USERNAME"),
        password=os.environ.get("MQTT_PASSWORD"),
        topic_root=os.environ.get(
            "LUCID_LANGSAM_TOPIC", "lucid/agents/langsam/components/langsam"
        ),
        default_prompt=prompt,
        timeout=60.0,
        debug_dir="./langsam_debug",
    )

    print(f"Image : {image_path}  ({img.shape[1]}x{img.shape[0]})")
    print(f"Prompt: {prompt!r}")
    print("Sending request...")

    mask = seg.segment(img, prompt=prompt)
    seg.close()

    total = mask.size
    hits = int(mask.sum())
    print(f"Result: {hits}/{total} pixels ({100 * hits / total:.1f}%)")
    if hits == 0:
        print("WARNING: empty mask — object not detected or server timed out.")
    else:
        print("Debug images saved to ./langsam_debug/")

if __name__ == "__main__":
    main()
