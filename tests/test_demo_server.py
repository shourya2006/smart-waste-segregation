"""Exercise the installed Gradio server and its serialized callback contract."""

import importlib.util
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("gradio"), "install the demo dependencies")
class DemoServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}"
        cls.log = tempfile.TemporaryFile(mode="w+")
        cls.process = subprocess.Popen(
            [sys.executable, "app.py", "--synthetic-fixture", "--port", str(port)],
            cwd=cls.root, stdout=cls.log, stderr=subprocess.STDOUT,
        )
        cls.addClassCleanup(cls.stop_server)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(cls.url + "/config", timeout=1) as response:
                    cls.config = json.load(response)
                return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.1)
        cls.log.seek(0)
        raise AssertionError("Real Gradio server failed to start: " + cls.log.read())

    @classmethod
    def stop_server(cls):
        if cls.process.poll() is None:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait(timeout=5)
        cls.log.close()

    def test_real_http_server_serves_interface(self):
        self.assertEqual(self.config["version"], "6.30.0")
        self.assertTrue(any(c["type"] == "image" for c in self.config["components"]))
        with urllib.request.urlopen(self.url, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("gradio", response.read().decode().lower())

    def test_upload_prediction_error_and_clear_api_contracts(self):
        from gradio_client import Client, handle_file

        client = Client(self.url, verbose=False)
        error = client.predict(None, api_name="/classify")
        self.assertIn("No image uploaded", error[0])
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "generated.png"
            Image.new("RGB", (48, 40), (200, 120, 60)).save(image)
            result = client.predict(handle_file(image), api_name="/classify")
        self.assertEqual(result[1], "Metal")
        self.assertEqual(len(result[4]["data"]), 6)
        cleared = client.predict(api_name="/clear")
        self.assertEqual(len(cleared), 7)
        self.assertIsNone(cleared[0])
        self.assertEqual(cleared[2], "")
        self.assertEqual(cleared[5]["data"], [])
