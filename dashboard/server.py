"""
FastAPI Mission Control Server & Edge Streaming Engine.
Provides real-time video streaming over IP (MJPEG/RTSP), WebSocket telemetry feeds,
and REST control APIs for the astronaut on-board cockpit and ground monitoring.
"""

import os
import cv2
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


class DashboardServer:
    """
    Edge web server and video streamer.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        pipeline_controller = None
    ):
        self.host = host
        self.port = port
        self.controller = pipeline_controller

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.templates_dir = os.path.join(base_dir, "templates")
        self.static_dir = os.path.join(base_dir, "static")

        self.app = FastAPI(title="Astro-HAR Edge Mission Control")
        self.templates = Jinja2Templates(directory=self.templates_dir)
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

        self.active_websockets: List[WebSocket] = []
        self._register_routes()

    def _register_routes(self):
        app = self.app

        @app.get("/", response_class=HTMLResponse)
        async def get_dashboard(request: Request):
            return self.templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "mission_name": "BHARATIYA ANTARIKSH STATION - IVA MONITOR",
                    "unit_id": "EDGE-PAYLOAD-AI-01"
                }
            )

        @app.get("/stream/video_feed")
        def video_feed():
            """Streams live processed frames over HTTP/IP to any connected device."""
            return StreamingResponse(
                self._generate_video_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @app.websocket("/ws/telemetry")
        async def websocket_telemetry(websocket: WebSocket):
            await websocket.accept()
            self.active_websockets.append(websocket)
            try:
                while True:
                    # Keep alive and receive any client ping
                    data = await websocket.receive_text()
            except WebSocketDisconnect:
                if websocket in self.active_websockets:
                    self.active_websockets.remove(websocket)

        @app.get("/api/state")
        def get_state():
            if self.controller:
                return self.controller.get_current_state()
            return {"status": "NO_CONTROLLER"}

        @app.post("/api/control/inject_anomaly")
        def inject_anomaly():
            if self.controller:
                self.controller.inject_out_of_sequence_anomaly()
                return {"status": "SUCCESS", "message": "Test out-of-sequence anomaly triggered."}
            return {"status": "ERROR"}

        @app.post("/api/control/inject_out_of_sequence")
        def inject_out_of_sequence():
            if self.controller:
                self.controller.inject_out_of_sequence_anomaly()
                return {"status": "SUCCESS", "message": "Out-of-sequence anomaly triggered."}
            return {"status": "ERROR"}

        @app.post("/api/control/inject_skipped_step")
        def inject_skipped_step():
            if self.controller:
                self.controller.inject_skipped_step_anomaly()
                return {"status": "SUCCESS", "message": "Skipped-step anomaly triggered."}
            return {"status": "ERROR"}

        @app.post("/api/control/next_step")
        def force_next_step():
            if self.controller:
                self.controller.force_advance_step()
                return {"status": "SUCCESS", "message": "Step forced to next."}
            return {"status": "ERROR"}

        @app.post("/api/control/reset")
        def reset_protocol():
            if self.controller:
                self.controller.reset_protocol()
                return {"status": "SUCCESS", "message": "Protocol reset."}
            return {"status": "ERROR"}

        @app.post("/api/control/toggle_audio")
        def toggle_audio():
            if self.controller:
                new_state = self.controller.toggle_audio()
                return {"status": "SUCCESS", "audio_enabled": new_state}
            return {"status": "ERROR"}

        @app.get("/api/telemetry/download")
        def download_telemetry():
            if self.controller and self.controller.telemetry_logger:
                path = self.controller.telemetry_logger.current_json_path
                if os.path.exists(path):
                    return FileResponse(path, media_type="application/json", filename=os.path.basename(path))
            return {"error": "Telemetry JSON file not available"}

        @app.get("/api/telemetry/text_content")
        def get_telemetry_text_content():
            """Returns the live formatted structured text report as text for the in-GUI viewer."""
            if self.controller and self.controller.telemetry_logger:
                content = self.controller.telemetry_logger.get_latest_text_content()
                filename = os.path.basename(self.controller.telemetry_logger.current_txt_path)
                return {"status": "SUCCESS", "filename": filename, "content": content}
            return {"status": "ERROR", "content": "Telemetry log not initialized yet."}

        @app.post("/api/control/switch_protocol")
        def switch_protocol(request_data: Dict[str, Any]):
            """Switches the active experiment protocol configuration."""
            protocol_name = request_data.get("protocol", "experiment_protocol")
            filename = f"{protocol_name}.yaml" if not protocol_name.endswith(".yaml") else protocol_name
            full_path = os.path.join("configs", filename)
            if self.controller:
                res = self.controller.switch_protocol(full_path)
                return res
            return {"status": "ERROR", "message": "No active controller"}

        @app.get("/api/telemetry/download_text")
        def download_telemetry_text():
            """Downloads timestamped structured lightweight text report (.txt)."""
            if self.controller and self.controller.telemetry_logger:
                path = self.controller.telemetry_logger.current_txt_path
                if os.path.exists(path):
                    return FileResponse(path, media_type="text/plain", filename=os.path.basename(path))
            return {"error": "Telemetry text file not available"}

        @app.get("/api/video/download")
        def download_video():
            """Downloads locally recorded experiment session MP4 video."""
            if self.controller and self.controller.video_recorder:
                path = self.controller.video_recorder.current_video_path
                if path and os.path.exists(path):
                    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))
            return {"error": "Video file not available"}

        @app.get("/api/video/export")
        def export_video():
            if self.controller and self.controller.video_recorder:
                path = self.controller.video_recorder.current_video_path
                if path and os.path.exists(path):
                    return {"status": "READY", "file_path": path, "filename": os.path.basename(path)}
            return {"status": "RECORDING", "message": "Video recording in progress."}

    def _generate_video_stream(self):
        """Yields MJPEG stream chunks."""
        while self.controller and self.controller.is_running:
            try:
                frame = self.controller.get_latest_annotated_frame()
                if frame is not None:
                    # Encode to JPEG
                    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                time.sleep(0.033)  # ~30 FPS network stream
            except (GeneratorExit, asyncio.CancelledError):
                break
            except Exception:
                break

    async def broadcast_telemetry(self, telemetry_data: Dict[str, Any]):
        """Pushes telemetry updates to all connected browser clients."""
        if not self.active_websockets:
            return

        msg = json.dumps(telemetry_data)
        disconnected = []
        for ws in self.active_websockets:
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            if ws in self.active_websockets:
                self.active_websockets.remove(ws)
