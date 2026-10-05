"""
Orientation-Agnostic Spatial Engine.
Performs edge object detection, hand tracking, and hand-object interaction (HOI) analysis.
In microgravity, astronaut orientation is arbitrary. All positions are mapped into
the payload rack canonical coordinate frame.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .payload_calibration import PayloadRackCalibrator
from .hmr_pose_tracker import HMRPoseTracker


class DetectedObject:
    def __init__(self, label: str, bbox: Tuple[int, int, int, int], confidence: float):
        self.label = label
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.center_px = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.center_canonical: Tuple[float, float] = (0.5, 0.5)


class SpatialEngine:
    """
    Combines object detection, hand proximity, and spatial geometric tracking.
    Lightweight, fast, and optimized for edge devices (Jetson, Raspberry Pi, laptop CPU).
    Includes orientation-agnostic 3D Human Mesh Recovery (HMR) relative to payload rack.
    """

    def __init__(self, calibrator: PayloadRackCalibrator):
        self.calibrator = calibrator
        self.hmr_tracker = HMRPoseTracker(calibrator=calibrator)
        self.prev_hand_pos: Optional[Tuple[float, float]] = None
        self.prev_time: float = 0.0

        # Colors for visualization (BGR)
        self.label_colors = {
            "main_box": (200, 180, 50),     # Cyan-gold
            "red_box": (50, 50, 230),       # Bright Red
            "yellow_box": (20, 220, 240),   # Bright Yellow
            "payload_slot": (180, 120, 20), # Deep Cyan
            "astronaut_hand": (100, 240, 100), # Green
            "specimen_sample": (220, 100, 220) # Magenta
        }

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        simulated_entities: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[DetectedObject], Dict[str, Any], np.ndarray]:
        """
        Detects objects & hands, transforms to canonical frame, computes HOI metrics.
        Returns: (detected_objects, spatial_context, annotated_frame)
        """
        h, w = frame.shape[:2]
        detected_objects: List[DetectedObject] = []

        if simulated_entities:
            # If provided simulated or synthetic metadata (e.g. from synthetic stream)
            for ent in simulated_entities:
                label = ent["label"]
                bbox = ent["bbox"]
                conf = ent.get("confidence", 0.92)
                obj = DetectedObject(label, bbox, conf)
                obj.center_canonical = self.calibrator.to_canonical_coords(obj.center_px)
                detected_objects.append(obj)
        else:
            # Live detection via robust computer vision color-geometry segmenters
            # for the Red Box, Yellow Box, Main Box, and Hand
            detected_objects = self._detect_experiment_objects(frame)

        # Map all objects to canonical coordinates
        for obj in detected_objects:
            obj.center_canonical = self.calibrator.to_canonical_coords(obj.center_px)

        # Identify hands and target objects
        hands = [o for o in detected_objects if o.label == "astronaut_hand"]
        active_objects = [o for o in detected_objects if o.label != "astronaut_hand"]

        # Compute Hand Velocity in canonical coordinates
        hand_velocity = (0.0, 0.0)
        hand_pos_canonical = (0.5, 0.5)
        hand_speed = 0.0

        if hands:
            primary_hand = hands[0]
            hand_pos_canonical = primary_hand.center_canonical
            if self.prev_hand_pos is not None and self.prev_time > 0:
                dt = max(timestamp - self.prev_time, 0.001)
                vx = (hand_pos_canonical[0] - self.prev_hand_pos[0]) / dt
                vy = (hand_pos_canonical[1] - self.prev_hand_pos[1]) / dt
                hand_velocity = (vx, vy)
                hand_speed = float(np.sqrt(vx * vx + vy * vy))
            self.prev_hand_pos = hand_pos_canonical
            self.prev_time = timestamp

        # Determine Hand-Object Interaction (HOI) Proximity
        closest_target = None
        min_dist_norm = 999.0
        for obj in active_objects:
            dx = obj.center_canonical[0] - hand_pos_canonical[0]
            dy = obj.center_canonical[1] - hand_pos_canonical[1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            if dist < min_dist_norm:
                min_dist_norm = dist
                closest_target = obj.label

        # Proximity threshold in normalized coordinates
        is_in_contact = (min_dist_norm <= 0.22 and closest_target is not None)
        contact_target = closest_target if is_in_contact else "none"

        # Check box containment status (whether red/yellow box are outside main box)
        main_box = next((o for o in active_objects if o.label == "main_box"), None)
        red_box = next((o for o in active_objects if o.label == "red_box"), None)
        yellow_box = next((o for o in active_objects if o.label == "yellow_box"), None)

        red_outside = False
        yellow_outside = False
        if main_box:
            mb_x1, mb_y1, mb_x2, mb_y2 = main_box.bbox
            if red_box:
                rx, ry = red_box.center_px
                # If center is outside main box bounds
                red_outside = not (mb_x1 <= rx <= mb_x2 and mb_y1 <= ry <= mb_y2)
            if yellow_box:
                yx, yy = yellow_box.center_px
                yellow_outside = not (mb_x1 <= yx <= mb_x2 and mb_y1 <= yy <= mb_y2)

        spatial_context = {
            "hand_contact_target": contact_target,
            "min_dist_norm": min_dist_norm if closest_target else 1.0,
            "hand_speed": hand_speed,
            "hand_velocity": hand_velocity,
            "hand_pos_canonical": hand_pos_canonical,
            "red_box_outside": red_outside,
            "yellow_box_outside": yellow_outside,
            "main_box_present": main_box is not None,
            "main_box_open": (red_outside or yellow_outside or contact_target == "main_box"),
            "active_objects_count": len(active_objects)
        }

        # 3D Human Mesh Recovery (HMR) & Posture Tracking relative to rack
        primary_hand = hands[0] if hands else None
        hmr_data = self.hmr_tracker.estimate_rack_relative_pose(primary_hand, spatial_context, (h, w))
        spatial_context["hmr_pose"] = {
            "roll_deg": hmr_data.get("roll_deg", 0.0),
            "pitch_deg": hmr_data.get("pitch_deg", 0.0),
            "yaw_deg": hmr_data.get("yaw_deg", 0.0),
            "reach_angle_deg": hmr_data.get("reach_angle_deg", 0.0),
            "stability_score": hmr_data.get("stability_score", 0.9),
            "mesh_confidence": hmr_data.get("mesh_confidence", 0.9),
            "orientation_label": hmr_data.get("orientation_label", "RACK_ALIGNED")
        }

        # Build feature vector for the Space-HAR Transformer (48 dimensions)
        feature_vector = self._extract_transformer_feature_vector(
            hand_pos_canonical, hand_velocity, active_objects, spatial_context
        )
        spatial_context["feature_vector"] = feature_vector

        # Render visual HUD overlay on frame (including 3D wireframe mesh)
        annotated_frame = self._render_spatial_hud(frame, detected_objects, spatial_context, hmr_data)

        return detected_objects, spatial_context, annotated_frame

    def _extract_transformer_feature_vector(
        self,
        hand_pos: Tuple[float, float],
        hand_vel: Tuple[float, float],
        objects: List[DetectedObject],
        context: Dict[str, Any]
    ) -> np.ndarray:
        """Encodes canonical spatial positions into fixed 48-D feature vector."""
        vec = np.zeros(48, dtype=np.float32)
        vec[0:2] = hand_pos
        vec[2:4] = hand_vel
        vec[4] = context.get("hand_speed", 0.0)
        vec[5] = context.get("min_dist_norm", 1.0)
        vec[6] = 1.0 if context.get("red_box_outside") else 0.0
        vec[7] = 1.0 if context.get("yellow_box_outside") else 0.0
        vec[8] = 1.0 if context.get("main_box_open") else 0.0

        # Object coordinates relative to hand
        target_map = {"main_box": 10, "red_box": 15, "yellow_box": 20, "payload_slot": 25}
        for obj in objects:
            idx = target_map.get(obj.label)
            if idx is not None:
                vec[idx:idx+2] = obj.center_canonical
                vec[idx+2] = obj.center_canonical[0] - hand_pos[0]
                vec[idx+3] = obj.center_canonical[1] - hand_pos[1]
                vec[idx+4] = obj.confidence

        return vec

    def _detect_experiment_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """Fast edge color/contour detection fallback for live camera feeds."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        detected = []

        # Color masks
        # 1. Red (handles wrap around 180)
        mask_r1 = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255]))
        mask_red = mask_r1 | mask_r2

        # 2. Yellow
        mask_yellow = cv2.inRange(hsv, np.array([18, 90, 90]), np.array([35, 255, 255]))

        # 3. Main Box (Blue/Grey container)
        mask_box = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([135, 255, 220]))

        # 4. Skin/Hand
        mask_skin = cv2.inRange(hsv, np.array([0, 25, 60]), np.array([25, 180, 255]))

        # Helper to find largest contour
        def find_largest(mask, min_area=1200):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                return (x, y, x + bw, y + bh)
            return None

        # Red box detection
        r_box = find_largest(mask_red, min_area=800)
        if r_box:
            detected.append(DetectedObject("red_box", r_box, 0.88))

        # Yellow box detection
        y_box = find_largest(mask_yellow, min_area=800)
        if y_box:
            detected.append(DetectedObject("yellow_box", y_box, 0.88))

        # Main box
        m_box = find_largest(mask_box, min_area=4000)
        if m_box:
            detected.append(DetectedObject("main_box", m_box, 0.82))
        else:
            # Default payload bay location in center
            default_mb = (int(w * 0.25), int(h * 0.30), int(w * 0.75), int(h * 0.85))
            detected.append(DetectedObject("main_box", default_mb, 0.70))

        # Hand detection
        h_box = find_largest(mask_skin, min_area=1500)
        if h_box:
            detected.append(DetectedObject("astronaut_hand", h_box, 0.85))

        return detected

    def _render_spatial_hud(
        self,
        frame: np.ndarray,
        objects: List[DetectedObject],
        context: Dict[str, Any],
        hmr_data: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Renders futuristic Space Station HUD with bounding boxes, interaction vectors, and 3D HMR wireframe."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Draw Payload Rack Calibration Boundary Grid
        if self.calibrator.rack_corners_px is not None:
            pts = self.calibrator.rack_corners_px.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=(100, 100, 100), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(
                overlay, "PAYLOAD CANONICAL RACK FRAME [0.0 - 1.0]",
                (int(self.calibrator.rack_corners_px[0][0]) + 10, int(self.calibrator.rack_corners_px[0][1]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA
            )

        # Draw Object Bounding Boxes
        for obj in objects:
            color = self.label_colors.get(obj.label, (255, 255, 255))
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            
            # Corner accents
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Tag label
            tag = f"{obj.label.upper()} [{obj.confidence:.2f}]"
            cx_can, cy_can = obj.center_canonical
            tag_coords = f"Rack:({cx_can:.2f},{cy_can:.2f})"

            cv2.putText(overlay, tag, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(overlay, tag_coords, (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        # Draw Hand-to-Target Interaction Vector line
        hands = [o for o in objects if o.label == "astronaut_hand"]
        if hands and context.get("hand_contact_target") != "none":
            hand = hands[0]
            target_label = context["hand_contact_target"]
            target = next((o for o in objects if o.label == target_label), None)
            if target:
                hx, hy = int(hand.center_px[0]), int(hand.center_px[1])
                tx, ty = int(target.center_px[0]), int(target.center_px[1])
                
                # Dynamic vector
                cv2.line(overlay, (hx, hy), (tx, ty), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(overlay, (hx, hy), 6, (0, 255, 0), -1)
                cv2.circle(overlay, (tx, ty), 6, (0, 255, 255), -1)

                mid_x, mid_y = (hx + tx) // 2, (hy + ty) // 2
                cv2.putText(
                    overlay, f"HOI: {target_label.upper()} (dist={context['min_dist_norm']:.2f})",
                    (mid_x - 40, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA
                )

        # Draw Orientation-Agnostic 3D Human Mesh Recovery (HMR) Wireframe
        if hmr_data is not None:
            overlay = self.hmr_tracker.render_hmr_wireframe(overlay, hmr_data)

        # Blend HUD overlay
        alpha = 0.90
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame
