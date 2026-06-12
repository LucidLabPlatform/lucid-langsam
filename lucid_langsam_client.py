#!/usr/bin/env python3
"""
LUCID LangSAM client.

Runs on any consumer box (e.g. the Kinova/ROS Python 3.8 machine). Requests
segmentation from the `langsam` component on Central Command over MQTT, using
the LUCID component command/result convention, and returns a boolean mask.

  publishes : <topic_root>/cmd/segment
  subscribes: <topic_root>/evt/segment/result

Drop-in replacement for a local LangSAM call:

    from lucid_langsam_client import LangSamClient

    seg = LangSamClient.from_env(default_prompt="watter bottle.")
    mask = seg.segment(bgr_image)            # -> bool (H, W) numpy array

Environment variables used by `from_env`:
    MQTT_HOST            (required)
    MQTT_PORT            (default 1883)
    MQTT_USERNAME        (optional)
    MQTT_PASSWORD        (optional)
    LUCID_LANGSAM_TOPIC  (default lucid/agents/central/components/langsam)

Python 3.8 compatible. Requires: paho-mqtt, numpy, opencv-python.
"""

import os
import json
import time
import uuid
import base64
import threading
from datetime import datetime

import numpy as np
import cv2

try:
    import paho.mqtt.client as mqtt
except ImportError as exc:  # pragma: no cover
    raise ImportError("LangSamClient requires paho-mqtt: pip install paho-mqtt") from exc


def _make_client(client_id):
    # Works on both paho-mqtt 1.x and 2.x.
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
    except (AttributeError, TypeError):
        return mqtt.Client(client_id=client_id)


class LangSamClient:
    """Remote LangSAM over MQTT. One persistent connection, blocking per call."""

    def __init__(
        self,
        host,
        port=1883,
        username=None,
        password=None,
        topic_root="lucid/agents/central/components/langsam",
        default_prompt="object.",
        timeout=120.0,
        jpeg_quality=90,
        debug_dir=None,
        client_id=None,
    ):
        self.cmd_topic = "{}/cmd/segment".format(topic_root)
        self.evt_topic = "{}/evt/segment/result".format(topic_root)
        self.default_prompt = default_prompt
        self.timeout = float(timeout)
        self.jpeg_quality = int(jpeg_quality)
        self.debug_dir = debug_dir

        self._responses = {}
        self._lock = threading.Lock()
        self._event = threading.Event()

        cid = client_id or "lucid-langsam-client-{}".format(os.getpid())
        self._mqtt = _make_client(cid)
        if username:
            self._mqtt.username_pw_set(username, password)
        self._mqtt.on_message = self._on_message
        self._mqtt.connect(host, int(port), keepalive=60)
        self._mqtt.subscribe(self.evt_topic, qos=1)
        self._mqtt.loop_start()

    @classmethod
    def from_env(cls, **overrides):
        kwargs = dict(
            host=os.environ["MQTT_HOST"],
            port=int(os.environ.get("MQTT_PORT", "1883")),
            username=os.environ.get("MQTT_USERNAME"),
            password=os.environ.get("MQTT_PASSWORD"),
            topic_root=os.environ.get(
                "LUCID_LANGSAM_TOPIC", "lucid/agents/central/components/langsam"
            ),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return
        rid = data.get("request_id")
        if rid is None:
            return
        with self._lock:
            self._responses[rid] = data
        self._event.set()

    def segment(self, bgr_image, prompt=None):
        """Return a boolean (H, W) segmap. Empty mask on timeout/error."""
        if prompt is None:
            prompt = self.default_prompt
        height, width = bgr_image.shape[:2]

        # JPEG keeps the payload under the broker's default 1 MB packet size.
        ok, buf = cv2.imencode(
            ".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            print("[langsam] failed to JPEG-encode frame")
            return np.zeros((height, width), dtype=bool)

        request_id = str(uuid.uuid4())
        payload = json.dumps(
            {
                "request_id": request_id,
                "prompt": prompt,
                "image_format": "jpg",
                "image_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
            }
        )
        self._mqtt.publish(self.cmd_topic, payload, qos=1)

        data = self._wait_for(request_id)
        if data is None:
            print("[langsam] remote segmentation timed out")
            return np.zeros((height, width), dtype=bool)
        if not data.get("ok"):
            print("[langsam] remote error: {}".format(data.get("error")))
            return np.zeros((height, width), dtype=bool)

        mask_bytes = base64.b64decode(data["mask_b64"])
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("[langsam] unreadable mask in response")
            return np.zeros((height, width), dtype=bool)

        segmap = mask > 0
        if segmap.shape != (height, width):
            segmap = cv2.resize(
                segmap.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if self.debug_dir:
            self._save_debug(bgr_image, segmap)
        return segmap

    def _wait_for(self, request_id):
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            with self._lock:
                data = self._responses.pop(request_id, None)
            if data is not None:
                return data
            self._event.wait(timeout=0.5)
            self._event.clear()
        return None

    def _save_debug(self, bgr_image, segmap):
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(
                os.path.join(self.debug_dir, "langsam_mask_{}.png".format(ts)),
                (segmap.astype(np.uint8) * 255),
            )
            overlay = bgr_image.copy()
            overlay[segmap] = (
                0.35 * overlay[segmap] + 0.65 * np.array([0, 255, 0])
            ).astype(np.uint8)
            cv2.imwrite(
                os.path.join(self.debug_dir, "langsam_overlay_{}.png".format(ts)),
                overlay,
            )
        except Exception as exc:
            print("[langsam] debug save failed: {}".format(exc))

    def close(self):
        try:
            self._mqtt.loop_stop()
            self._mqtt.disconnect()
        except Exception:
            pass
