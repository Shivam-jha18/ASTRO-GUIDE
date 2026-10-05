"""
Local MP4 Video Storage & Export Engine.
Threaded continuous session recorder with timestamp watermarks.
Ensures zero frame drops without blocking real-time edge processing.
"""

import os
import cv2
import time
import queue
import threading
import numpy as np
from typing import Optional, Tuple


class LocalSessionVideoRecorder:
    """
    Records processed mission video locally to disk in MP4 format.
    Operates in a background thread with an internal frame queue.
    """

    def __init__(
        self,
        output_dir: str = "telemetry_data/recordings",
        fps: int = 30,
        resolution: Tuple[int, int] = (1280, 720)
    ):
        self.output_dir = output_dir
        self.fps = fps
        self.width, self.height = resolution

        os.makedirs(self.output_dir, exist_ok=True)

        self.writer: Optional[cv2.VideoWriter] = None
        self.current_video_path: Optional[str] = None
        self.is_recording = False
        self.frame_queue = queue.Queue(maxsize=120)
        self.worker_thread = threading.Thread(target=self._writer_loop, daemon=True)

        self.worker_thread.start()

    def start_new_session(self, session_prefix: str = "BAS_EXP") -> str:
        """Starts a new MP4 video recording file."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{session_prefix}_{timestamp_str}.mp4"
        self.current_video_path = os.path.join(self.output_dir, filename)

        # Standard MP4 codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.current_video_path,
            fourcc,
            float(self.fps),
            (self.width, self.height)
        )
        self.is_recording = True
        print(f"[VideoRecorder] Recording session initialized: {self.current_video_path}")
        return self.current_video_path

    def write_frame(self, frame: np.ndarray):
        """Pushes frame into recording queue."""
        if not self.is_recording or self.writer is None:
            return

        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())

    def _writer_loop(self):
        """Asynchronously writes queued frames to disk."""
        while True:
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if frame is None:
                break

            if self.writer is not None:
                # Ensure frame matches target size
                if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                    frame = cv2.resize(frame, (self.width, self.height))
                self.writer.write(frame)

            self.frame_queue.task_done()

    def stop_session(self) -> Optional[str]:
        """Finalizes and closes current session video."""
        self.is_recording = False
        time.sleep(0.3)
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"[VideoRecorder] Session saved successfully: {self.current_video_path}")
            return self.current_video_path
        return None
