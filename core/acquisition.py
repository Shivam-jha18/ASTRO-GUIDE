"""
Frame Acquisition & Pre-processing Unit (60 FPS capable).
Decoupled multi-threaded frame grabber with support for payload UVC cameras,
RTSP/IP video feeds, local MP4 files, and synthetic microgravity simulation feeds.
"""

import cv2
import time
import math
import threading
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


class FrameAcquisitionUnit:
    """
    Threaded acquisition engine ensuring zero-latency frame capture.
    Maintains a single latest-frame buffer to eliminate camera queue buffering lag.
    """

    def __init__(
        self,
        source: Any = 0,
        target_resolution: Tuple[int, int] = (1280, 720),
        target_fps: int = 60,
        synthetic_fallback: bool = True
    ):
        self.source = source
        self.target_w, self.target_h = target_resolution
        self.target_fps = target_fps
        self.synthetic_fallback = synthetic_fallback

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_synthetic = False

        self.current_frame: Optional[np.ndarray] = None
        self.current_timestamp: float = 0.0
        self.simulated_entities: List[Dict[str, Any]] = []

        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

        # Simulation state machine for synthetic feed
        self.sim_time: float = 0.0
        self.sim_forced_anomaly: Optional[str] = None
        self.sim_speed: float = 1.0

        self._init_source()

    def _init_source(self):
        """Attempts to open physical/file source or falls back to synthetic feed."""
        opened = False
        if not self.synthetic_fallback or self.source != "synthetic":
            try:
                self.cap = cv2.VideoCapture(self.source)
                if self.cap is not None and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
                    self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    # Test read
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        opened = True
                        print(f"[Acquisition] Hardware / Stream camera '{self.source}' successfully connected.")
            except Exception as e:
                print(f"[Acquisition] Failed to open source '{self.source}': {e}")

        if not opened:
            print("[Acquisition] Activating Synthetic Microgravity Simulation Feed (Bharatiya Antariksh Station Payload).")
            self.is_synthetic = True

    def start(self):
        """Starts background acquisition thread."""
        self.is_running = True
        self.thread.start()

    def stop(self):
        """Stops capture thread and releases camera."""
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float, List[Dict[str, Any]]]:
        """Returns the most recent frame, timestamp, and optional simulated entity metadata."""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy(), self.current_timestamp, list(self.simulated_entities)
            return None, 0.0, []

    def trigger_simulation_anomaly(self, anomaly_action: str = "EXTRACT_YELLOW_EARLY"):
        """Triggers a test out-of-order execution in the synthetic simulator."""
        self.sim_forced_anomaly = anomaly_action
        print(f"[Acquisition Simulation] INJECTING ANOMALY: {anomaly_action}")

    def _capture_loop(self):
        """Continuous capture loop running at target FPS."""
        frame_interval = 1.0 / self.target_fps

        while self.is_running:
            t0 = time.time()

            if not self.is_synthetic and self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    if (frame.shape[1], frame.shape[0]) != (self.target_w, self.target_h):
                        frame = cv2.resize(frame, (self.target_w, self.target_h))
                    with self.lock:
                        self.current_frame = frame
                        self.current_timestamp = time.time()
                        self.simulated_entities = []
                else:
                    # If video file ended, loop back
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                # Generate synthetic microgravity experiment frame
                frame, entities = self._render_synthetic_frame()
                with self.lock:
                    self.current_frame = frame
                    self.current_timestamp = time.time()
                    self.simulated_entities = entities

            # Sleep to match target FPS
            elapsed = time.time() - t0
            sleep_time = max(frame_interval - elapsed, 0.001)
            time.sleep(sleep_time)

    def _render_synthetic_frame(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Synthesizes a realistic 60 FPS payload camera view inside the Space Station rack.
        Simulates microgravity floating physics and astronaut hands manipulating
        the Main Box, Red Box, and Yellow Box.
        """
        w, h = self.target_w, self.target_h
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 1. Background: Futuristic Space Station IVA Payload Rack (metallic gradient + grid)
        # Deep space station interior background
        rack_bg = np.linspace(25, 45, h, dtype=np.uint8)[:, np.newaxis]
        frame[:, :, 0] = rack_bg
        frame[:, :, 1] = rack_bg + 5
        frame[:, :, 2] = rack_bg + 10

        # Draw structural payload rack frame
        cv2.rectangle(frame, (int(w * 0.12), int(h * 0.12)), (int(w * 0.88), int(h * 0.88)), (60, 70, 80), 2)
        cv2.line(frame, (int(w * 0.12), int(h * 0.22)), (int(w * 0.88), int(h * 0.22)), (50, 60, 70), 1)

        # Microgravity subtle floating oscillation
        self.sim_time += 0.025 * self.sim_speed
        t = self.sim_time
        drift_x = math.sin(t * 0.8) * 8.0
        drift_y = math.cos(t * 0.6) * 5.0

        entities: List[Dict[str, Any]] = []

        # 2. Main Containment Box (Mounted on payload rack shelf)
        mb_x1 = int(w * 0.32 + drift_x * 0.3)
        mb_y1 = int(h * 0.36 + drift_y * 0.3)
        mb_x2 = int(w * 0.68 + drift_x * 0.3)
        mb_y2 = int(h * 0.80 + drift_y * 0.3)

        # Main box chassis
        cv2.rectangle(frame, (mb_x1, mb_y1), (mb_x2, mb_y2), (80, 85, 95), -1)
        cv2.rectangle(frame, (mb_x1, mb_y1), (mb_x2, mb_y2), (180, 190, 200), 2)
        cv2.putText(
            frame, "BAS PAYLOAD EXP-26174",
            (mb_x1 + 15, mb_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA
        )

        entities.append({
            "label": "main_box",
            "bbox": (mb_x1, mb_y1, mb_x2, mb_y2),
            "confidence": 0.94
        })

        # Payload Slot on right side of rack
        slot_x1, slot_y1 = int(w * 0.72), int(h * 0.38)
        slot_x2, slot_y2 = int(w * 0.85), int(h * 0.58)
        cv2.rectangle(frame, (slot_x1, slot_y1), (slot_x2, slot_y2), (40, 50, 60), -1)
        cv2.rectangle(frame, (slot_x1, slot_y1), (slot_x2, slot_y2), (0, 200, 200), 1)
        cv2.putText(frame, "PAYLOAD SLOT", (slot_x1 + 5, slot_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 200), 1)

        entities.append({
            "label": "payload_slot",
            "bbox": (slot_x1, slot_y1, slot_x2, slot_y2),
            "confidence": 0.90
        })

        # 3. Microgravity Experiment Step Cycle (Simulated Autonomous Timeline)
        # Cycle duration: 55 seconds (or looped)
        cycle_len = 50.0
        phase = t % cycle_len

        # Red & Yellow Box base positions inside main box
        red_base_x = mb_x1 + 40
        red_base_y = mb_y1 + 50
        yellow_base_x = mb_x1 + 180
        yellow_base_y = mb_y1 + 50
        box_w, box_h = 90, 75

        # Check if anomaly injected:
        if self.sim_forced_anomaly == "EXTRACT_YELLOW_EARLY":
            # Force yellow extraction prematurely in Phase 1
            yellow_pos_x = int(w * 0.50 + math.sin(t * 2) * 20)
            yellow_pos_y = int(h * 0.30 + math.cos(t * 2) * 15)
            red_pos_x = red_base_x
            red_pos_y = red_base_y
            hand_x = yellow_pos_x + 30
            hand_y = yellow_pos_y + 30
        else:
            # Normal sequential experiment timeline
            if phase < 7.0:
                # Step 1: Astronaut hand approaches and unlatches/opens main box
                hand_x = int(mb_x1 + (phase / 7.0) * (mb_x2 - mb_x1) * 0.5)
                hand_y = int(mb_y1 - 40 + math.sin(phase * 3) * 15)
                red_pos_x, red_pos_y = red_base_x, red_base_y
                yellow_pos_x, yellow_pos_y = yellow_base_x, yellow_base_y

            elif phase < 15.0:
                # Step 2: Extract Red Box
                progress = (phase - 7.0) / 8.0
                red_pos_x = int(red_base_x + progress * 80)
                red_pos_y = int(red_base_y - progress * 130 + math.sin(phase * 2) * 6)
                yellow_pos_x, yellow_pos_y = yellow_base_x, yellow_base_y
                hand_x = red_pos_x + 40
                hand_y = red_pos_y + 40

            elif phase < 22.0:
                # Step 3: Inspect Red Box specimen
                red_pos_x = int(w * 0.45 + math.sin(phase * 1.5) * 8)
                red_pos_y = int(h * 0.28 + math.cos(phase * 1.2) * 6)
                yellow_pos_x, yellow_pos_y = yellow_base_x, yellow_base_y
                hand_x = red_pos_x + 45
                hand_y = red_pos_y + 35

            elif phase < 30.0:
                # Step 4: Extract Yellow Box
                progress = (phase - 22.0) / 8.0
                red_pos_x = int(w * 0.25)
                red_pos_y = int(h * 0.28)
                yellow_pos_x = int(yellow_base_x + progress * 40)
                yellow_pos_y = int(yellow_base_y - progress * 110)
                hand_x = yellow_pos_x + 40
                hand_y = yellow_pos_y + 40

            elif phase < 38.0:
                # Step 5: Perform Interaction in Payload Slot
                red_pos_x = int(w * 0.25)
                red_pos_y = int(h * 0.28)
                progress = (phase - 30.0) / 8.0
                yellow_pos_x = int(slot_x1 - 20 + math.sin(phase * 3) * 5)
                yellow_pos_y = int(slot_y1 + 10)
                hand_x = yellow_pos_x + 40
                hand_y = yellow_pos_y + 30

            elif phase < 45.0:
                # Step 6: Return both boxes into main containment
                progress = (phase - 38.0) / 7.0
                red_pos_x = int(red_base_x + (1.0 - progress) * 80)
                red_pos_y = int(red_base_y + (1.0 - progress) * (-100))
                yellow_pos_x = int(yellow_base_x + (1.0 - progress) * 100)
                yellow_pos_y = int(yellow_base_y + (1.0 - progress) * (-80))
                hand_x = yellow_pos_x + 40
                hand_y = yellow_pos_y + 30

            else:
                # Step 7: Close & lock main box
                red_pos_x, red_pos_y = red_base_x, red_base_y
                yellow_pos_x, yellow_pos_y = yellow_base_x, yellow_base_y
                hand_x = int(mb_x1 + 100 + math.sin(phase * 4) * 20)
                hand_y = int(mb_y1 - 10)

        # 4. Render Red Specimen Box
        rx1, ry1 = int(red_pos_x), int(red_pos_y)
        rx2, ry2 = rx1 + box_w, ry1 + box_h
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (30, 30, 220), -1)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (80, 80, 255), 2)
        cv2.putText(frame, "RED BOX", (rx1 + 10, ry1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        entities.append({
            "label": "red_box",
            "bbox": (rx1, ry1, rx2, ry2),
            "confidence": 0.95
        })

        # 5. Render Yellow Reagent Box
        yx1, ry1 = int(yellow_pos_x), int(yellow_pos_y)
        yx2, ry2 = yx1 + box_w, ry1 + box_h
        cv2.rectangle(frame, (yx1, ry1), (yx2, ry2), (20, 210, 240), -1)
        cv2.rectangle(frame, (yx1, ry1), (yx2, ry2), (80, 230, 255), 2)
        cv2.putText(frame, "YELLOW BOX", (yx1 + 6, ry1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)

        entities.append({
            "label": "yellow_box",
            "bbox": (yx1, ry1, yx2, ry2),
            "confidence": 0.95
        })

        # 6. Render Astronaut Hand (EVA/IVA Glove with sensors)
        hx1, hy1 = int(hand_x - 35), int(hand_y - 35)
        hx2, hy2 = int(hand_x + 35), int(hand_y + 35)
        cv2.circle(frame, (int(hand_x), int(hand_y)), 30, (80, 180, 240), -1)
        cv2.circle(frame, (int(hand_x), int(hand_y)), 30, (200, 255, 255), 2)
        cv2.circle(frame, (int(hand_x), int(hand_y)), 6, (0, 255, 0), -1)
        cv2.putText(frame, "ASTRONAUT GLOVE", (hx1 - 10, hy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 255, 100), 1)

        entities.append({
            "label": "astronaut_hand",
            "bbox": (hx1, hy1, hx2, hy2),
            "confidence": 0.92
        })

        # Timestamp and Unit ID Watermark
        cv2.putText(
            frame, f"FEED: SYNTHETIC 0-G SIMULATOR | {time.strftime('%Y-%m-%d %H:%M:%S')}",
            (int(w * 0.13), int(h * 0.17)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA
        )

        return frame, entities
