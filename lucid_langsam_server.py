#!/usr/bin/env python3
"""
LUCID LangSAM GPU server.

Runs on the Central Command machine (the only GPU box). Loads LangSAM ONCE
(warm, on CUDA) and answers segmentation requests over MQTT, following the
LUCID component command/result convention:

  subscribe : lucid/agents/<AGENT_ID>/components/langsam/cmd/segment
  publish   : lucid/agents/<AGENT_ID>/components/langsam/evt/segment/result
  status    : lucid/agents/<AGENT_ID>/components/langsam/status   (retained)

Request payload (JSON):
  {
    "request_id": "<uuid>",
    "prompt": "water bottle.",
    "image_format": "jpg" | "png",
    "image_b64": "<base64 of encoded image bytes>"
  }

Result payload (JSON):
  {
    "request_id": "<uuid>",
    "ok": true,
    "mask_format": "png",
    "mask_b64": "<base64 of 8-bit PNG, 0/255>",
    "height": <int>, "width": <int>,
    "num_detections": <int>,
    "error": null
  }

Run inside the isolated LangSAM env (CUDA torch + lang_sam installed):

  MQTT_HOST=<broker> MQTT_PORT=1883 \
  MQTT_USERNAME=<user> MQTT_PASSWORD=<pass> \
  LUCID_LANGSAM_AGENT_ID=central \
  python lucid_langsam_server.py

This is a standalone responder that intentionally speaks the SAME topic
contract as a real lucid component, so it can later be promoted into a proper
`lucid-component-langsam` (cmd/<action> -> evt/<action>/result) with no change
to the Kinova client.
"""

import os
import json
import base64
import logging
import threading
from collections import OrderedDict

import numpy as np
import cv2
from PIL import Image
import paho.mqtt.client as mqtt

LOG = logging.getLogger("lucid.langsam.server")

AGENT_ID = os.environ.get("LUCID_LANGSAM_AGENT_ID", "central")
COMPONENT_ID = "langsam"
TOPIC_ROOT = os.environ.get(
    "LUCID_LANGSAM_TOPIC",
    f"lucid/agents/{AGENT_ID}/components/{COMPONENT_ID}",
)
CMD_TOPIC = f"{TOPIC_ROOT}/cmd/segment"
EVT_TOPIC = f"{TOPIC_ROOT}/evt/segment/result"
STATUS_TOPIC = f"{TOPIC_ROOT}/status"

MQTT_HOST = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USERNAME = os.environ.get("MQTT_USERNAME")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD")

DEFAULT_PROMPT = os.environ.get("LUCID_LANGSAM_PROMPT", "object.")


class Segmenter:
    """Wraps LangSAM. The model is loaded once and kept warm on the GPU."""

    def __init__(self):
        # Import here so the heavy model libs load only in this process.
        from lang_sam import LangSAM

        # NOTE: match this constructor to YOUR installed lang-segment-anything.
        # Newer builds accept e.g. LangSAM(sam_type="sam2.1_hiera_small").
        self.model = LangSAM()

        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                LOG.warning("CUDA not available - LangSAM is running on CPU!")
        except Exception:
            self.device = "unknown"
        LOG.info("LangSAM loaded (device=%s)", self.device)

    def segment(self, bgr_image, prompt):
        """Return (bool (H, W) mask, num_detections). Mask = union of detections."""
        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb)

        # ---- LangSAM call: adapt to your installed version ----------------
        # Newer API (lang-segment-anything >= 0.2.x):
        results = self.model.predict([image_pil], [prompt])
        masks = self._extract_masks(results, h, w)
        # Older API (<= 0.1.x):
        #   masks_t, boxes, phrases, logits = self.model.predict(image_pil, prompt)
        #   masks = (masks_t.cpu().numpy().astype(bool)
        #            if len(masks_t) else np.zeros((0, h, w), bool))
        # -------------------------------------------------------------------

        if masks.shape[0] == 0:
            return np.zeros((h, w), dtype=bool), 0
        union = np.any(masks, axis=0)
        return union.astype(bool), int(masks.shape[0])

    @staticmethod
    def _extract_masks(results, h, w):
        """Normalize possible return shapes into an (N, H, W) bool array."""
        if not results:
            return np.zeros((0, h, w), bool)
        r0 = results[0]
        masks = r0.get("masks") if isinstance(r0, dict) else None
        if masks is None:
            return np.zeros((0, h, w), bool)
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        return masks.astype(bool)


def _make_client(client_id):
    # Works on both paho-mqtt 1.x and 2.x.
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
    except (AttributeError, TypeError):
        return mqtt.Client(client_id=client_id)


class LangSamServer:
    def __init__(self):
        self._seg = Segmenter()
        self._gpu_lock = threading.Lock()      # serialize GPU work
        self._seen = OrderedDict()             # request_id dedup (single net thread)
        self._seen_max = 256

        self._client = _make_client(f"lucid-langsam-{os.getpid()}")
        if MQTT_USERNAME:
            self._client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        # Last-will: if we drop off, mark the component as error (retained).
        self._client.will_set(
            STATUS_TOPIC, json.dumps({"state": "error"}), qos=1, retain=True
        )

    def _is_duplicate(self, rid):
        if not rid:
            return False
        if rid in self._seen:
            return True
        self._seen[rid] = True
        if len(self._seen) > self._seen_max:
            self._seen.popitem(last=False)
        return False

    def _on_connect(self, client, userdata, flags, rc, *args):
        LOG.info("Connected to MQTT (rc=%s). Subscribing: %s", rc, CMD_TOPIC)
        client.subscribe(CMD_TOPIC, qos=1)
        client.publish(
            STATUS_TOPIC, json.dumps({"state": "running"}), qos=1, retain=True
        )

    def _on_message(self, client, userdata, msg):
        try:
            req = json.loads(msg.payload.decode("utf-8"))
        except Exception as exc:
            LOG.warning("Bad request payload: %s", exc)
            return
        rid = req.get("request_id", "")
        if self._is_duplicate(rid):
            LOG.info("Duplicate request_id %s ignored", rid)
            return
        # Offload so the network loop stays responsive; GPU is serialized below.
        threading.Thread(target=self._handle, args=(req, rid), daemon=True).start()

    def _handle(self, req, rid):
        prompt = req.get("prompt") or DEFAULT_PROMPT
        try:
            img_bytes = base64.b64decode(req["image_b64"])
            arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise ValueError("could not decode request image")

            with self._gpu_lock:
                mask, n = self._seg.segment(arr, prompt)

            ok, png = cv2.imencode(".png", (mask.astype(np.uint8) * 255))
            if not ok:
                raise ValueError("could not encode mask PNG")

            h, w = mask.shape[:2]
            result = {
                "request_id": rid,
                "ok": True,
                "mask_format": "png",
                "mask_b64": base64.b64encode(png.tobytes()).decode("ascii"),
                "height": int(h),
                "width": int(w),
                "num_detections": n,
                "error": None,
            }
            LOG.info("Request %s: %d detections, %d mask px", rid, n, int(mask.sum()))
        except Exception as exc:
            LOG.exception("Segmentation failed for request %s", rid)
            result = {
                "request_id": rid,
                "ok": False,
                "mask_format": "png",
                "mask_b64": None,
                "error": str(exc),
            }
        self._client.publish(EVT_TOPIC, json.dumps(result), qos=1)

    def run(self):
        self._client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        self._client.loop_forever()


def main():
    logging.basicConfig(
        level=os.environ.get("LUCID_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LangSamServer().run()


if __name__ == "__main__":
    main()
