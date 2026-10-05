"""
Dynamic Finite State Machine & Protocol Verifier.
The mission-critical reasoning engine for validating scientific experiment SOPs
in space (Bharatiya Antariksh Station / Lunar IVA).
Verifies sequence order, detects skipped/out-of-order steps, triggers voice alerts,
and generates structured lightweight telemetry logs.
"""

import time
import yaml
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    VIOLATED = "VIOLATED"


class AnomalyType(str, Enum):
    OUT_OF_SEQUENCE = "OUT_OF_SEQUENCE"
    STEP_SKIPPED = "STEP_SKIPPED"
    TIMEOUT_EXCEEDED = "TIMEOUT_EXCEEDED"
    UNAUTHORIZED_OBJECT = "UNAUTHORIZED_OBJECT"
    REVERSE_REGRESSION = "REVERSE_REGRESSION"


class ProtocolStepRecord(BaseModel):
    id: int
    code: str
    name: str
    description: str
    target_object: str
    expected_action: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_sec: float = 0.0
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0


class FSMProtocolVerifier:
    """
    Finite State Machine verifying compliance with Standard Operating Procedures (SOP).
    Enforces deterministic validation over neural network action predictions.
    """

    def __init__(
        self,
        protocol_config_path: str,
        on_step_advanced: Optional[Callable[[ProtocolStepRecord], None]] = None,
        on_alert_triggered: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_voice_prompt: Optional[Callable[[str], None]] = None,
    ):
        self.config_path = protocol_config_path
        self.on_step_advanced = on_step_advanced
        self.on_alert_triggered = on_alert_triggered
        self.on_voice_prompt = on_voice_prompt

        self.protocol_name = "UNKNOWN"
        self.protocol_title = "Scientific Experiment"
        self.steps_data: List[Dict[str, Any]] = []
        self.steps: List[ProtocolStepRecord] = []
        self.current_step_index: int = 0
        self.experiment_start_time: Optional[float] = None
        self.experiment_end_time: Optional[float] = None
        self.is_completed: bool = False

        # Anti-jitter detection smoothing
        self.active_detection_count: int = 0
        self.min_consecutive_frames_to_advance: int = 15  # ~0.5s at 30 FPS
        self.last_alert_time: float = 0.0
        self.alert_cooldown_sec: float = 4.0  # Prevent spamming alerts

        self.load_protocol(protocol_config_path)

    def load_protocol(self, config_path: str):
        """Loads and parses protocol definition YAML."""
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.protocol_name = data.get("protocol_name", "EXP_26174")
        self.protocol_title = data.get("protocol_title", "Microgravity Experiment")
        self.completion_message = data.get("completion_message", "Experiment completed.")
        self.steps_data = data.get("steps", [])

        self.steps = []
        for s in self.steps_data:
            self.steps.append(
                ProtocolStepRecord(
                    id=s["id"],
                    code=s["code"],
                    name=s["name"],
                    description=s["description"],
                    target_object=s["target_object"],
                    expected_action=s["expected_action"],
                    status=StepStatus.PENDING
                )
            )

        self.reset()

    def reset(self):
        """Resets the state machine to the initial state."""
        self.current_step_index = 0
        self.experiment_start_time = None
        self.experiment_end_time = None
        self.is_completed = False
        self.active_detection_count = 0
        self.last_alert_time = 0.0

        for s in self.steps:
            s.status = StepStatus.PENDING
            s.start_time = None
            s.end_time = None
            s.duration_sec = 0.0
            s.anomalies = []

        # Start initial step
        if self.steps:
            self.steps[0].status = StepStatus.IN_PROGRESS
            self.steps[0].start_time = time.time()
            self.experiment_start_time = time.time()

            # Trigger initial voice prompt
            initial_prompt = self.steps_data[0].get("voice_prompt")
            if initial_prompt and self.on_voice_prompt:
                self.on_voice_prompt(initial_prompt)

    @property
    def current_step(self) -> Optional[ProtocolStepRecord]:
        """Returns the currently active step, or None if completed."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def next_suggested_step(self) -> Optional[Dict[str, Any]]:
        """Returns metadata for the expected next step."""
        if self.current_step_index + 1 < len(self.steps_data):
            return self.steps_data[self.current_step_index + 1]
        return None

    def update(
        self,
        predicted_action: str,
        target_object: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Processes current perception input and checks against FSM protocol rules.
        """
        now = time.time()
        curr = self.current_step
        if curr is None or self.is_completed:
            return {"status": "COMPLETE", "step": None, "alert": None}

        # Initialize start time if needed
        if curr.start_time is None:
            curr.start_time = now

        curr_raw = self.steps_data[self.current_step_index]
        expected_act = curr_raw["expected_action"]
        expected_obj = curr_raw["target_object"]

        # Check for potential out-of-order execution or skipped steps
        # Look ahead across remaining steps to see if astronaut jumped ahead
        for future_idx in range(self.current_step_index + 1, len(self.steps_data)):
            future_step = self.steps_data[future_idx]
            if (
                predicted_action == future_step["expected_action"] and
                target_object == future_step["target_object"] and
                confidence >= 0.70
            ):
                # If jumped more than 1 step ahead, it's a STEP_SKIPPED anomaly!
                is_skip = (future_idx > self.current_step_index + 1)
                anomaly_type = AnomalyType.STEP_SKIPPED if is_skip else AnomalyType.OUT_OF_SEQUENCE

                if is_skip:
                    msg = f"Step skipped! Attempting '{future_step['name']}' before completing '{curr.name}'."
                    voice_msg = (
                        f"Alert! Protocol step skipped. You must complete {curr.name} before starting {future_step['name']}."
                    )
                else:
                    msg = f"Out of order action detected! Expected '{curr.name}', but detected '{future_step['name']}'."
                    voice_msg = future_step.get(
                        "voice_alert_skip",
                        f"Alert! Out of order step detected. Please complete {curr.name} first."
                    )

                alert_info = self._trigger_anomaly(
                    anomaly_type=anomaly_type,
                    message=msg,
                    voice_text=voice_msg,
                    detected_step=future_step["code"],
                    expected_step=curr.code
                )
                return {"status": "ANOMALY", "step": curr, "alert": alert_info}

        # Check if current expected step action is being satisfied
        action_match = (predicted_action == expected_act)
        object_match = (target_object == expected_obj or expected_obj == "any")

        if action_match and object_match and confidence >= 0.50:
            self.active_detection_count += 1
            curr.confidence = max(curr.confidence, confidence)

            # Advance after sustained detection threshold
            if self.active_detection_count >= self.min_consecutive_frames_to_advance:
                self._advance_step(now)
                return {"status": "ADVANCED", "step": curr, "alert": None}
        else:
            # Decay counter smoothly
            if self.active_detection_count > 0:
                self.active_detection_count -= 1

        # Check for timeout / prolonged inactivity
        timeout_threshold = curr_raw.get("timeout_sec", 60.0)
        time_elapsed = now - curr.start_time
        if time_elapsed > timeout_threshold and (now - self.last_alert_time) > self.alert_cooldown_sec:
            prompt = curr_raw.get("voice_prompt", f"Reminder: Please execute {curr.name}.")
            self._trigger_voice(prompt)
            self.last_alert_time = now

        return {
            "status": "TRACKING",
            "current_step": curr,
            "progress_percent": int(((self.current_step_index) / len(self.steps)) * 100),
            "alert": None
        }

    def _advance_step(self, timestamp: float):
        """Marks current step as completed and transitions to the next step."""
        curr = self.current_step
        if curr is None:
            return

        curr.status = StepStatus.COMPLETED
        curr.end_time = timestamp
        curr.duration_sec = round(timestamp - (curr.start_time or timestamp), 2)
        self.active_detection_count = 0

        if self.on_step_advanced:
            self.on_step_advanced(curr)

        # Move to next step
        self.current_step_index += 1
        if self.current_step_index < len(self.steps):
            next_step = self.steps[self.current_step_index]
            next_step.status = StepStatus.IN_PROGRESS
            next_step.start_time = timestamp
            
            # Trigger voice prompt for the new active step
            prompt = self.steps_data[self.current_step_index].get("voice_prompt")
            if prompt:
                self._trigger_voice(prompt)
        else:
            # All steps completed!
            self.is_completed = True
            self.experiment_end_time = timestamp
            if self.on_voice_prompt:
                self.on_voice_prompt(self.completion_message)

    def force_advance_step(self):
        """Manual operator override to advance step from GUI."""
        self._advance_step(time.time())

    def _trigger_anomaly(
        self,
        anomaly_type: AnomalyType,
        message: str,
        voice_text: str,
        detected_step: str,
        expected_step: str
    ) -> Optional[Dict[str, Any]]:
        """Handles and logs an anomaly event."""
        now = time.time()
        if (now - self.last_alert_time) < self.alert_cooldown_sec:
            return None  # Cooldown active

        self.last_alert_time = now
        alert_payload = {
            "timestamp": now,
            "type": anomaly_type.value,
            "message": message,
            "voice_alert": voice_text,
            "detected_step": detected_step,
            "expected_step": expected_step
        }

        curr = self.current_step
        if curr:
            curr.anomalies.append(alert_payload)

        # Trigger audio alert
        self._trigger_voice(voice_text)

        # Trigger callback
        if self.on_alert_triggered:
            self.on_alert_triggered(alert_payload)

        return alert_payload

    def _trigger_voice(self, text: str):
        """Dispatches voice announcement."""
        if self.on_voice_prompt:
            self.on_voice_prompt(text)

    def get_structured_telemetry(self) -> Dict[str, Any]:
        """
        Produces the structured lightweight telemetry dictionary for JSON export.
        """
        total_duration = 0.0
        if self.experiment_start_time:
            end = self.experiment_end_time or time.time()
            total_duration = round(end - self.experiment_start_time, 2)

        return {
            "mission": "BHARATIYA_ANTARIKSH_STATION_IVA",
            "protocol_code": self.protocol_name,
            "protocol_title": self.protocol_title,
            "is_completed": self.is_completed,
            "total_duration_sec": total_duration,
            "current_step_id": self.current_step.id if self.current_step else len(self.steps),
            "next_suggested_step": self.next_suggested_step,
            "total_steps": len(self.steps),
            "completion_percentage": int((sum(1 for s in self.steps if s.status == StepStatus.COMPLETED) / len(self.steps)) * 100) if self.steps else 0,
            "steps": [s.model_dump() for s in self.steps]
        }

    def simulate_out_of_sequence_anomaly(self) -> Dict[str, Any]:
        """Triggers a simulated out-of-sequence step addition anomaly."""
        curr = self.current_step
        curr_name = curr.name if curr else "Active Step"
        alert = {
            "timestamp": time.time(),
            "type": AnomalyType.OUT_OF_SEQUENCE.value,
            "message": f"Out of sequence action detected! Extraneous action performed before completing '{curr_name}'.",
            "voice_alert": f"Alert! Out of sequence action detected. Please complete {curr_name} first.",
            "detected_step": "STEP_OUT_OF_SEQUENCE",
            "expected_step": curr.code if curr else "STEP_1_OPEN_MAIN_BOX"
        }
        if curr:
            curr.anomalies.append(alert)
        self._trigger_voice(alert["voice_alert"])
        if self.on_alert_triggered:
            self.on_alert_triggered(alert)
        return alert

    def simulate_skipped_step_anomaly(self) -> Dict[str, Any]:
        """Triggers a simulated skipped step anomaly."""
        curr = self.current_step
        curr_name = curr.name if curr else "Step 1"
        next_step = self.next_suggested_step
        next_name = next_step["name"] if next_step else "Next Step"
        alert = {
            "timestamp": time.time(),
            "type": AnomalyType.STEP_SKIPPED.value,
            "message": f"Critical step skipped! Astronaut jumped ahead to '{next_name}' while '{curr_name}' was incomplete.",
            "voice_alert": f"Alert! Critical step skipped. You must complete {curr_name} before proceeding.",
            "detected_step": next_step["code"] if next_step else "STEP_SKIPPED",
            "expected_step": curr.code if curr else "STEP_1_OPEN_MAIN_BOX"
        }
        if curr:
            curr.anomalies.append(alert)
        self._trigger_voice(alert["voice_alert"])
        if self.on_alert_triggered:
            self.on_alert_triggered(alert)
        return alert
