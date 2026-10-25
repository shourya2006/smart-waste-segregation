"""Capture demo screenshots for desktop and mobile layouts using headless Chrome."""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 7860
URL = f"http://127.0.0.1:{PORT}"
OUT_DIR = Path("docs")

def wait_for_server(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(URL, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def main():
    print(f"Launching app.py on port {PORT} with synthetic fixture...")
    proc = subprocess.Popen(
        [sys.executable, "app.py", "--synthetic-fixture", "--port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        if not wait_for_server():
            print("Timed out waiting for Gradio server to become ready.")
            out, err = proc.communicate(timeout=2)
            print("STDOUT:", out.decode())
            print("STDERR:", err.decode())
            sys.exit(1)
        print("Server is ready! Capturing desktop layout (1280x800)...")
        # Let Gradio frontend hydrate
        time.sleep(3)
        desktop_png = OUT_DIR / "demo-desktop.png"
        subprocess.run([
            CHROME_BIN,
            "--headless=new",
            "--disable-gpu",
            "--virtual-time-budget=5000",
            "--window-size=1280,800",
            f"--screenshot={desktop_png}",
            URL,
        ], check=True)
        print(f"Desktop screenshot saved to {desktop_png} ({desktop_png.stat().st_size} bytes)")

        print("Capturing narrow-screen / mobile layout (390x844)...")
        mobile_png = OUT_DIR / "demo-mobile.png"
        subprocess.run([
            CHROME_BIN,
            "--headless=new",
            "--disable-gpu",
            "--virtual-time-budget=5000",
            "--window-size=390,844",
            f"--screenshot={mobile_png}",
            URL,
        ], check=True)
        print(f"Mobile screenshot saved to {mobile_png} ({mobile_png.stat().st_size} bytes)")

    finally:
        print("Terminating demo server...")
        proc.terminate()
        proc.wait(timeout=5)
        print("Done.")

if __name__ == "__main__":
    main()
