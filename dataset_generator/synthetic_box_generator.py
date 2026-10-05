"""
Synthetic Microgravity Dataset Generator for Dual-Containment Payload Experiment.
Generates thousands of synthetic, annotated training frames with random 6-DOF rotations,
simulating microgravity floating behavior without requiring actual zero-G flight data.
Outputs YOLO format annotations: <class_id> <x_center> <y_center> <width> <height>.
"""

import os
import cv2
import math
import random
import numpy as np
from typing import Tuple, List, Dict, Any


class SyntheticDatasetGenerator:
    """
    Synthesizes training images and YOLO annotation files for:
    0: main_box
    1: red_box
    2: yellow_box
    3: astronaut_hand
    """

    def __init__(
        self,
        output_dir: str = "dataset/synthetic_microgravity",
        image_size: Tuple[int, int] = (640, 640)
    ):
        self.output_dir = output_dir
        self.img_w, self.img_h = image_size
        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.classes = {
            "main_box": 0,
            "red_box": 1,
            "yellow_box": 2,
            "astronaut_hand": 3
        }

    def generate_batch(self, num_samples: int = 100, split: str = "train"):
        """Generates a batch of synthetic microgravity training samples."""
        split_img_dir = os.path.join(self.images_dir, split)
        split_lbl_dir = os.path.join(self.labels_dir, split)
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        print(f"[DatasetGenerator] Generating {num_samples} synthetic samples in {split} set...")

        for idx in range(num_samples):
            img, annotations = self.generate_single_frame()
            base_name = f"synth_bas_{split}_{idx:05d}"

            # Save Image
            img_path = os.path.join(split_img_dir, f"{base_name}.jpg")
            cv2.imwrite(img_path, img)

            # Save YOLO Annotation
            lbl_path = os.path.join(split_lbl_dir, f"{base_name}.txt")
            with open(lbl_path, "w") as f:
                for ann in annotations:
                    f.write(f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n")

        print(f"[DatasetGenerator] Complete. Dataset ready at {self.output_dir}")

    def generate_single_frame(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Synthesizes a single frame with realistic random microgravity orientation."""
        # 1. Background: Metallic space station rack texture with random lighting gradient
        img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        base_gray = random.randint(20, 55)
        img[:] = (base_gray, base_gray + random.randint(0, 8), base_gray + random.randint(2, 12))

        # Add random subtle rack lines / grid
        for _ in range(random.randint(2, 6)):
            gx = random.randint(0, self.img_w)
            cv2.line(img, (gx, 0), (gx, self.img_h), (base_gray + 15, base_gray + 18, base_gray + 20), 1)

        annotations: List[Dict[str, Any]] = []

        # 2. Main Box (Blue/Grey chassis) with random scale and rotation
        mb_w = random.randint(260, 360)
        mb_h = random.randint(180, 260)
        mb_cx = random.randint(int(self.img_w * 0.35), int(self.img_w * 0.65))
        mb_cy = random.randint(int(self.img_h * 0.35), int(self.img_h * 0.70))
        angle_deg = random.uniform(-35.0, 35.0)  # Microgravity tilt

        mb_mask, mb_pts = self._draw_rotated_box(
            img, (mb_cx, mb_cy), (mb_w, mb_h), angle_deg,
            fill_color=(80, 85, 95), border_color=(190, 200, 210)
        )
        annotations.append(self._to_yolo_bbox(self.classes["main_box"], mb_pts))

        # 3. Red Box (either inside main box or floating in 0-G)
        red_outside = random.random() > 0.45
        rb_w, rb_h = random.randint(65, 95), random.randint(55, 80)
        if red_outside:
            rb_cx = random.randint(int(self.img_w * 0.15), int(self.img_w * 0.85))
            rb_cy = random.randint(int(self.img_h * 0.15), int(self.img_h * 0.50))
            rb_angle = random.uniform(-180.0, 180.0)  # Full 360 floating rotation
        else:
            rb_cx = mb_cx - int(mb_w * 0.22)
            rb_cy = mb_cy
            rb_angle = angle_deg + random.uniform(-10.0, 10.0)

        _, rb_pts = self._draw_rotated_box(
            img, (rb_cx, rb_cy), (rb_w, rb_h), rb_angle,
            fill_color=(35, 35, random.randint(190, 240)),
            border_color=(90, 90, 255)
        )
        annotations.append(self._to_yolo_bbox(self.classes["red_box"], rb_pts))

        # 4. Yellow Box
        yellow_outside = random.random() > 0.60
        yb_w, yb_h = random.randint(65, 95), random.randint(55, 80)
        if yellow_outside:
            yb_cx = random.randint(int(self.img_w * 0.15), int(self.img_w * 0.85))
            yb_cy = random.randint(int(self.img_h * 0.15), int(self.img_h * 0.55))
            yb_angle = random.uniform(-180.0, 180.0)
        else:
            yb_cx = mb_cx + int(mb_w * 0.22)
            yb_cy = mb_cy
            yb_angle = angle_deg + random.uniform(-10.0, 10.0)

        _, yb_pts = self._draw_rotated_box(
            img, (yb_cx, yb_cy), (yb_w, yb_h), yb_angle,
            fill_color=(random.randint(15, 30), random.randint(200, 245), random.randint(220, 255)),
            border_color=(80, 240, 255)
        )
        annotations.append(self._to_yolo_bbox(self.classes["yellow_box"], yb_pts))

        # 5. Astronaut Hand / EVA Glove (interacts with one of the items or floats)
        if random.random() > 0.20:
            hand_target = random.choice([rb_pts, yb_pts, mb_pts, None])
            if hand_target is not None:
                # Place hand near target box edge
                hx = int(np.mean(hand_target[:, 0]) + random.randint(-25, 25))
                hy = int(np.mean(hand_target[:, 1]) + random.randint(-25, 25))
            else:
                hx = random.randint(int(self.img_w * 0.2), int(self.img_w * 0.8))
                hy = random.randint(int(self.img_h * 0.2), int(self.img_h * 0.8))

            hradius = random.randint(28, 42)
            cv2.circle(img, (hx, hy), hradius, (80, 180, 230), -1)
            cv2.circle(img, (hx, hy), hradius, (200, 240, 255), 2)
            cv2.circle(img, (hx, hy), 6, (0, 255, 0), -1)

            hand_pts = np.array([
                [hx - hradius, hy - hradius],
                [hx + hradius, hy - hradius],
                [hx + hradius, hy + hradius],
                [hx - hradius, hy + hradius]
            ])
            annotations.append(self._to_yolo_bbox(self.classes["astronaut_hand"], hand_pts))

        # Add camera sensor noise & microgravity lighting gradient
        noise = np.random.normal(0, random.uniform(3, 8), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img, annotations

    def _draw_rotated_box(
        self,
        img: np.ndarray,
        center: Tuple[int, int],
        size: Tuple[int, int],
        angle_deg: float,
        fill_color: Tuple[int, int, int],
        border_color: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draws a 2D oriented bounding box with arbitrary microgravity rotation."""
        rect = ((center[0], center[1]), (size[0], size[1]), angle_deg)
        box = cv2.boxPoints(rect)
        pts = np.int32(box)
        cv2.fillPoly(img, [pts], fill_color)
        cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)
        return img, pts

    def _to_yolo_bbox(self, class_id: int, pts: np.ndarray) -> Dict[str, Any]:
        """Converts polygon corners to normalized YOLO <cx cy w h>."""
        x_min = max(float(np.min(pts[:, 0])), 0.0)
        x_max = min(float(np.max(pts[:, 0])), float(self.img_w))
        y_min = max(float(np.min(pts[:, 1])), 0.0)
        y_max = min(float(np.max(pts[:, 1])), float(self.img_h))

        w = (x_max - x_min) / self.img_w
        h = (y_max - y_min) / self.img_h
        cx = (x_min + x_max) / (2.0 * self.img_w)
        cy = (y_min + y_max) / (2.0 * self.img_h)

        return {
            "class_id": class_id,
            "cx": float(np.clip(cx, 0.0, 1.0)),
            "cy": float(np.clip(cy, 0.0, 1.0)),
            "w": float(np.clip(w, 0.01, 1.0)),
            "h": float(np.clip(h, 0.01, 1.0))
        }
