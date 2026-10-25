"""Capture demo screenshots for desktop and mobile layouts using headless Chrome."""

import argparse
import shutil
import socket
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-fixture", action="store_true")
    parser.add_argument("--browser-executable", type=Path)
    args = parser.parse_args()
    browser = args.browser_executable or shutil.which("chromium") or shutil.which("google-chrome")
    browser = str(browser or CHROME_BIN)
    if not Path(browser).is_file():
        parser.error("Provide --browser-executable with the path to Chrome or Chromium")
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", PORT))
        except OSError:
            parser.error(f"Port {PORT} is already in use; stop that server before capturing")
    command = [sys.executable, "app.py", "--port", str(PORT)]
    if args.synthetic_fixture:
        command.append("--synthetic-fixture")
    print(f"Launching {'synthetic fixture' if args.synthetic_fixture else 'evaluated-model'} demo")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(command)
    try:
        if not wait_for_server():
            print("Timed out waiting for Gradio server to become ready.")
            sys.exit(1)
        print("Server is ready! Capturing desktop layout (1280x800)...")
        # Let Gradio frontend hydrate
        time.sleep(3)
        desktop_png = OUT_DIR / "demo-desktop.png"
        subprocess.run([
            browser,
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
            browser,
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
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        print("Done.")

if __name__ == "__main__":
    main()
