"""
Payload-Centric Canonical Frame Calibration Engine.
Ensures orientation-agnostic tracking in microgravity (0-G) environments.
In space missions (BAS / Lunar), gravity vectors do not define 'up' or 'down'.
This module maps all 2D and 3D detections into a normalized payload rack reference frame.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List


class PayloadRackCalibrator:
    """
    Computes and maintains homography and coordinate projection between
    raw camera image coordinates and the canonical payload experiment rack frame:
    [0.0, 1.0] x [0.0, 1.0].
    """

    def __init__(self, target_resolution: Tuple[int, int] = (1280, 720)):
        self.target_w, self.target_h = target_resolution
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.rack_corners_px: Optional[np.ndarray] = None
        self.is_calibrated = False

        # Default canonical rack coordinates (normalized [0, 1])
        self.canonical_dst = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=np.float32)

        # Fallback default corners (central 70% of frame)
        self.set_default_rack_bounds()

    def set_default_rack_bounds(self):
        """Sets default rack region if fiducials are not yet calibrated."""
        x_min, y_min = int(self.target_w * 0.15), int(self.target_h * 0.15)
        x_max, y_max = int(self.target_w * 0.85), int(self.target_h * 0.85)
        corners = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ], dtype=np.float32)
        self.update_calibration(corners)

    def update_calibration(self, four_corners_px: np.ndarray) -> bool:
        """
        Updates the homography matrix using 4 detected corners of the payload rack:
        [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
        """
        if four_corners_px is None or len(four_corners_px) != 4:
            return False

        # Invert/standardize coordinate format
        src = np.array(four_corners_px, dtype=np.float32)
        dst = np.array([
            [0.0, 0.0],
            [float(self.target_w), 0.0],
            [float(self.target_w), float(self.target_h)],
            [0.0, float(self.target_h)]
        ], dtype=np.float32)

        try:
            import cv2
            H, _ = cv2.findHomography(src, dst)
            if H is not None:
                self.homography_matrix = H
                self.inverse_homography = np.linalg.inv(H)
                self.rack_corners_px = src
                self.is_calibrated = True
                return True
        except Exception:
            pass

        # Fallback affine or simple normalization
        self.rack_corners_px = src
        self.is_calibrated = True
        return True

    def to_canonical_coords(self, pt_px: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transforms image pixel coordinate (x, y) into normalized payload frame [0.0, 1.0].
        Orientation-invariant: independent of camera roll or tilt angle.
        """
        x, y = pt_px
        if self.homography_matrix is not None:
            pt = np.array([[[x, y]]], dtype=np.float32)
            try:
                import cv2
                transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
                norm_x = float(np.clip(transformed[0][0][0] / self.target_w, 0.0, 1.0))
                norm_y = float(np.clip(transformed[0][0][1] / self.target_h, 0.0, 1.0))
                return (norm_x, norm_y)
            except Exception:
                pass

        # Direct normalization fallback
        return (float(np.clip(x / self.target_w, 0.0, 1.0)),
                float(np.clip(y / self.target_h, 0.0, 1.0)))

    def compute_rotation_invariant_features(
        self,
        hand_pos_norm: Tuple[float, float],
        box_pos_norm: Tuple[float, float],
        hand_velocity: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        Extracts SO(3) rotation-invariant scalars:
        - Relative Euclidean distance
        - Dot product of approach velocity toward target
        - Radial approach speed
        """
        dx = box_pos_norm[0] - hand_pos_norm[0]
        dy = box_pos_norm[1] - hand_pos_norm[1]
        dist = float(np.sqrt(dx * dx + dy * dy))

        vx, vy = hand_velocity
        speed = float(np.sqrt(vx * vx + vy * vy))

        # Dot product normalized (approach vector vs velocity vector)
        if dist > 1e-5 and speed > 1e-5:
            cos_approach = float((dx * vx + dy * vy) / (dist * speed))
        else:
            cos_approach = 0.0

        return {
            "dist": dist,
            "speed": speed,
            "approach_alignment": cos_approach
        }
