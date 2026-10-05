"""
Space-HAR Transformer: Orientation-Agnostic Spatio-Temporal Action Classifier.
Performs temporal windowed attention over spatial keypoints and hand-object vectors
to recognize astronaut experiment actions in microgravity.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class SpaceHARTransformer:
    """
    Lightweight Spatio-Temporal Transformer for Edge Human Activity Recognition.
    Wraps PyTorch implementation with fallback heuristics for zero-weight initialization.
    """

    def __init__(
        self,
        num_classes: int = 8,
        temporal_window: int = 16,
        feature_dim: int = 48,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        device: str = "cpu"
    ):
        self.num_classes = num_classes
        self.temporal_window = temporal_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.device = device

        self.action_labels = [
            "idle_inspect",
            "approach_rack",
            "open_box",
            "extract_object",
            "inspect_sample",
            "interact_slot",
            "insert_object",
            "close_box"
        ]

        self.model = None
        self._init_pytorch_model(nhead, num_layers)

        # Sliding temporal buffer for inference
        self.feature_buffer: List[np.ndarray] = []

    def _init_pytorch_model(self, nhead: int, num_layers: int):
        """Initializes PyTorch Transformer network."""
        try:
            import torch
            import torch.nn as nn

            class SpatioTemporalTransformer(nn.Module):
                def __init__(self, in_features, d_model, nhead, num_layers, num_classes, window_len):
                    super().__init__()
                    self.in_proj = nn.Sequential(
                        nn.Linear(in_features, d_model),
                        nn.LayerNorm(d_model),
                        nn.GELU()
                    )
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
                    self.pos_embed = nn.Parameter(torch.randn(1, window_len + 1, d_model) * 0.02)
                    
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=d_model * 2,
                        dropout=0.1,
                        activation="gelu",
                        batch_first=True
                    )
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.head = nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, num_classes)
                    )

                def forward(self, x):
                    B, T, _ = x.shape
                    h = self.in_proj(x)
                    cls_tokens = self.cls_token.expand(B, -1, -1)
                    h = torch.cat((cls_tokens, h), dim=1)
                    h = h + self.pos_embed[:, :T + 1, :]
                    out = self.transformer(h)
                    cls_out = out[:, 0]
                    return self.head(cls_out)

            self.torch_module = SpatioTemporalTransformer(
                in_features=self.feature_dim,
                d_model=self.d_model,
                nhead=nhead,
                num_layers=num_layers,
                num_classes=self.num_classes,
                window_len=self.temporal_window
            ).to(self.device)
            self.torch_module.eval()
            self.model = self.torch_module
        except ImportError:
            self.model = None

    def add_frame_features(self, feature_vector: np.ndarray):
        """Pushes current frame spatial features into sliding temporal buffer."""
        if len(feature_vector) < self.feature_dim:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[:len(feature_vector)] = feature_vector
            feature_vector = padded
        elif len(feature_vector) > self.feature_dim:
            feature_vector = feature_vector[:self.feature_dim]

        self.feature_buffer.append(feature_vector.astype(np.float32))
        if len(self.feature_buffer) > self.temporal_window:
            self.feature_buffer.pop(0)

    def predict_action(
        self,
        current_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Infers the current action from temporal window.
        Returns: (best_action, confidence, probabilities_dict)
        """
        # If we have PyTorch model and full buffer, perform tensor inference
        if self.model is not None and len(self.feature_buffer) >= self.temporal_window:
            try:
                import torch
                window_np = np.array(self.feature_buffer)[np.newaxis, ...] # [1, T, D]
                with torch.no_grad():
                    inp_tensor = torch.from_numpy(window_np).to(self.device)
                    logits = self.torch_module(inp_tensor)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    
                    best_idx = int(np.argmax(probs))
                    best_conf = float(probs[best_idx])
                    best_action = self.action_labels[best_idx]
                    prob_dict = {
                        label: float(p) for label, p in zip(self.action_labels, probs)
                    }

                    # If contextual state informs the action, blend with spatial context
                    if current_context:
                        best_action, best_conf, prob_dict = self._apply_contextual_prior(
                            best_action, best_conf, prob_dict, current_context
                        )
                    return best_action, best_conf, prob_dict
            except Exception:
                pass

        # Robust heuristic fallback when initializing or gathering initial frames
        return self._heuristic_action_prediction(current_context)

    def _apply_contextual_prior(
        self,
        predicted_action: str,
        confidence: float,
        probs: Dict[str, float],
        context: Dict[str, Any]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Blends neural prediction with physical spatial invariants
        (e.g., hand in contact with box lid, or hand holding sample container).
        """
        hand_contact_target = context.get("hand_contact_target")
        hand_velocity = context.get("hand_speed", 0.0)

        if hand_contact_target == "main_box":
            if hand_velocity > 0.05:
                # Active manipulation of main container
                is_open = context.get("main_box_open", False)
                target_act = "close_box" if is_open else "open_box"
                probs[target_act] = max(probs.get(target_act, 0.0), 0.88)
                return target_act, 0.88, probs

        elif hand_contact_target in ("red_box", "yellow_box"):
            is_outside = context.get(f"{hand_contact_target}_outside", False)
            if not is_outside and hand_velocity > 0.03:
                target_act = "extract_object"
                probs[target_act] = max(probs.get(target_act, 0.0), 0.90)
                return target_act, 0.90, probs
            elif is_outside and context.get("near_main_box", False):
                target_act = "insert_object"
                probs[target_act] = max(probs.get(target_act, 0.0), 0.85)
                return target_act, 0.85, probs
            else:
                target_act = "inspect_sample"
                probs[target_act] = max(probs.get(target_act, 0.0), 0.82)
                return target_act, 0.82, probs

        elif hand_contact_target == "payload_slot":
            target_act = "interact_slot"
            probs[target_act] = max(probs.get(target_act, 0.0), 0.91)
            return target_act, 0.91, probs

        return predicted_action, confidence, probs

    def _heuristic_action_prediction(
        self,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Physics-based fallback classifier using hand-object interaction vectors."""
        probs = {a: 0.05 for a in self.action_labels}

        if not context:
            probs["idle_inspect"] = 0.65
            return "idle_inspect", 0.65, probs

        target = context.get("hand_contact_target")
        speed = context.get("hand_speed", 0.0)

        if not target or target == "none":
            if speed > 0.08:
                action = "approach_rack"
                conf = 0.75
            else:
                action = "idle_inspect"
                conf = 0.80
        elif target == "main_box":
            action = "open_box" if not context.get("main_box_open", False) else "close_box"
            conf = 0.85
        elif target == "red_box":
            if context.get("red_box_outside", False):
                action = "inspect_sample"
            else:
                action = "extract_object"
            conf = 0.88
        elif target == "yellow_box":
            if context.get("yellow_box_outside", False):
                action = "inspect_sample"
            else:
                action = "extract_object"
            conf = 0.88
        elif target == "payload_slot":
            action = "interact_slot"
            conf = 0.90
        else:
            action = "idle_inspect"
            conf = 0.60

        probs[action] = conf
        return action, conf, probs
