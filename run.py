#!/usr/bin/env python3
"""
Astro-HAR Quick Launcher (SIH 2026 - Problem 26174).
Starts the end-to-end edge pipeline and GUI dashboard server.
"""

import os
import sys
import argparse

# Ensure virtualenv python is used if invoked directly and available on the current OS
if sys.platform == "win32":
    venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe"))
else:
    venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "bin", "python"))

if sys.executable != venv_python and os.path.isfile(venv_python):
    try:
        os.execv(venv_python, [venv_python] + sys.argv)
    except OSError:
        pass

# Configure utf-8 console output for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yaml
from main import AstroHARPipelineController
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Astro-HAR Edge Mission Control Launcher")
    parser.add_argument("--camera", default=None, help="Camera index (e.g. 0) or video file path")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic microgravity simulation feed")
    parser.add_argument("--port", type=int, default=8080, help="Web dashboard and stream port")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind server to")
    parser.add_argument("--stream-ip", default=None, help="Target specific IP address for video stream broadcast")
    parser.add_argument("--no-audio", action="store_true", help="Disable voice alerts")
    args = parser.parse_args()

    # Load and adjust pipeline config if command line arguments provided
    config_path = "configs/pipeline_config.yaml"
    with open(config_path, "r") as f:
        pipe_cfg = yaml.safe_load(f)

    if args.synthetic:
        pipe_cfg["camera"]["source"] = "synthetic"
        pipe_cfg["camera"]["synthetic_fallback"] = True
    elif args.camera is not None:
        try:
            cam_idx = int(args.camera)
            pipe_cfg["camera"]["source"] = cam_idx
        except ValueError:
            pipe_cfg["camera"]["source"] = args.camera
        pipe_cfg["camera"]["synthetic_fallback"] = False

    if args.no_audio:
        pipe_cfg["audio"]["enabled"] = False

    bind_host = args.stream_ip if args.stream_ip else args.host
    pipe_cfg["server"]["host"] = bind_host
    pipe_cfg["server"]["port"] = args.port

    # Initialize controller with customized configuration
    controller = AstroHARPipelineController(pipe_cfg=pipe_cfg)
    controller.start()

    host = bind_host
    port = args.port

    # Detect local LAN IP for convenient display
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    # Auto-open browser on launch
    import webbrowser
    import threading
    def _open_browser():
        import time
        time.sleep(1.2)
        webbrowser.open(f"http://localhost:{port}")
    threading.Thread(target=_open_browser, daemon=True).start()

    print("\n" + "=" * 68)
    print(f"  🛰️  ASTRO-HAR IS LIVE & MONITORING EXPERIMENT SEQUENCE")
    print(f"  🌐  Local Cockpit HUD:  http://localhost:{port}")
    print(f"  📡  Station Stream IP:  http://{local_ip}:{port}/stream/video_feed")
    print(f"  📋  Protocol:           Dual-Containment (Red & Yellow Box)")
    print(f"  🧍  3D Pose / HMR:      Orientation-Agnostic Rack Reference Frame")
    print(f"  🔊  Voice Alerts:       {'DISABLED' if args.no_audio else 'ACTIVE (Offline)'}")
    print(f"  📄  Telemetry:          Structured JSON & Lightweight .txt Reports")
    print(f"  👉  Opening dashboard automatically in your browser...")
    print(f"  ⚠️  NOTE: KEEP THIS TERMINAL OPEN. (Press Ctrl+C to stop)")
    print("=" * 68 + "\n")
    try:
        config = uvicorn.Config(controller.server.app, host=host, port=port, log_level="error")
        server = uvicorn.Server(config)
        server.run()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        controller.stop()
        print("\n🛰️  Astro-HAR edge session ended cleanly. All telemetry and videos saved in telemetry_data/\n")


if __name__ == "__main__":
    main()
