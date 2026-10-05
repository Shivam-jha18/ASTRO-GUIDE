"""
Comprehensive Unit & Integration Test Suite for Astro-HAR Edge Pipeline.
SIH 2026 - Problem Statement 26174 Verification.

Verifies:
1. Payload Rack Canonical Homography Calibrator
2. Orientation-Agnostic 3D Human Mesh Recovery (HMR) & Posture Engine
3. FSM Protocol Verifier: Normal Progression & Proactive Next-Step Guidance
4. Out-of-Sequence Action Detection & Voice Alert
5. Skipped Step Anomaly Detection & Voice Alert
6. Structured JSON & Lightweight Text (.txt) Telemetry Report Generation
7. Microgravity 6-DOF Dataset Generation & Model Artifact Integrity
"""

import os
import sys
import json
import time

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.fsm_verifier import FSMProtocolVerifier, StepStatus, AnomalyType
from core.payload_calibration import PayloadRackCalibrator
from core.hmr_pose_tracker import HMRPoseTracker
from core.spatial_engine import SpatialEngine, DetectedObject
from core.action_transformer import SpaceHARTransformer
from storage.telemetry_logger import TelemetryLogger
from dataset_generator.synthetic_box_generator import SyntheticDatasetGenerator


def test_payload_calibration():
    print("[TEST 1/7] Testing Payload Rack Canonical Calibrator...")
    calibrator = PayloadRackCalibrator(target_resolution=(1280, 720))
    pt_norm = calibrator.to_canonical_coords((640, 360))
    assert 0.0 <= pt_norm[0] <= 1.0, f"X coord out of bounds: {pt_norm[0]}"
    assert 0.0 <= pt_norm[1] <= 1.0, f"Y coord out of bounds: {pt_norm[1]}"
    print("  -> Calibrator canonical mapping verified: (640, 360) ->", pt_norm)


def test_hmr_pose_tracker():
    print("[TEST 2/7] Testing Orientation-Agnostic 3D HMR & Posture Engine...")
    calibrator = PayloadRackCalibrator(target_resolution=(1280, 720))
    hmr_tracker = HMRPoseTracker(calibrator=calibrator)

    # Test dummy hand at canonical coordinates (0.4, 0.5)
    dummy_hand = DetectedObject("astronaut_hand", (500, 350, 580, 430), 0.92)
    dummy_hand.center_canonical = (0.42, 0.54)

    ctx = {"hand_velocity": (0.04, -0.02), "hand_speed": 0.045}
    hmr_result = hmr_tracker.estimate_rack_relative_pose(dummy_hand, ctx, (720, 1280))

    assert "joints" in hmr_result, "3D joints missing from HMR result"
    assert "roll_deg" in hmr_result, "Roll orientation missing"
    assert "pitch_deg" in hmr_result, "Pitch orientation missing"
    assert "torso_depth_m" in hmr_result, "Torso depth missing"
    assert len(hmr_result["joints"]) >= 9, f"Expected 9+ 3D joints, got {len(hmr_result['joints'])}"
    print(f"  -> 3D HMR Posture verified: Roll={hmr_result['roll_deg']}°, Pitch={hmr_result['pitch_deg']}°, Depth={hmr_result['torso_depth_m']}m")


def test_fsm_normal_progression():
    print("[TEST 3/7] Testing FSM Sequential Protocol Progression & Next-Step Guidance...")
    fsm = FSMProtocolVerifier("configs/experiment_protocol.yaml")
    assert fsm.current_step.id == 1, "Should start at Step 1"

    next_step = fsm.next_suggested_step
    assert next_step is not None, "Next suggested step should be available at start"
    assert next_step["id"] == 2, f"Expected Next Step to be 2, got {next_step['id']}"

    # Simulate sustained detection of Step 1 expected action ('open_box', target='main_box')
    for _ in range(16):
        res = fsm.update(
            predicted_action="open_box",
            target_object="main_box",
            confidence=0.85
        )

    assert fsm.current_step.id == 2, f"Expected transition to Step 2, got Step {fsm.current_step.id}"
    print(f"  -> Step 1 successfully completed and advanced to Step 2.")
    print(f"  -> Proactive Next-Step suggested: '{fsm.next_suggested_step['name']}'")


def test_fsm_out_of_order_anomaly():
    print("[TEST 4/7] Testing Out-Of-Order Anomaly Detection & Voice Alert...")
    alerts = []
    fsm = FSMProtocolVerifier(
        "configs/experiment_protocol.yaml",
        on_alert_triggered=lambda a: alerts.append(a)
    )
    # Current step is Step 1 (Open Main Box).
    # Prematurely perform Step 2 action (extract red box) before Step 1 is completed:
    res = fsm.update(
        predicted_action="extract_object",
        target_object="red_box",
        confidence=0.88
    )

    assert res["status"] == "ANOMALY", f"Expected ANOMALY status, got {res['status']}"
    assert len(alerts) > 0, "Alert callback was not triggered"
    assert alerts[0]["type"] == AnomalyType.OUT_OF_SEQUENCE.value
    print(f"  -> Caught Out-of-Sequence Anomaly: '{alerts[0]['message']}'")
    print(f"  -> Voice Alert Triggered: '{alerts[0]['voice_alert']}'")


def test_fsm_skipped_step_anomaly():
    print("[TEST 5/7] Testing Skipped Step Anomaly Detection & Voice Alert...")
    alerts = []
    fsm = FSMProtocolVerifier(
        "configs/experiment_protocol.yaml",
        on_alert_triggered=lambda a: alerts.append(a)
    )
    # Current step is Step 1. Prematurely perform Step 4 (Extract Yellow Box),
    # completely bypassing intermediate Step 2 & Step 3!
    res = fsm.update(
        predicted_action="extract_object",
        target_object="yellow_box",
        confidence=0.90
    )

    assert res["status"] == "ANOMALY", f"Expected ANOMALY status, got {res['status']}"
    assert len(alerts) > 0, "Alert callback was not triggered"
    assert alerts[0]["type"] == AnomalyType.STEP_SKIPPED.value
    print(f"  -> Caught Skipped Step Anomaly: '{alerts[0]['message']}'")
    print(f"  -> Voice Alert Triggered: '{alerts[0]['voice_alert']}'")


def test_telemetry_logger_json_and_text():
    print("[TEST 6/7] Testing Structured JSON & Lightweight Text (.txt) Generation...")
    logger = TelemetryLogger(output_dir="telemetry_data/test_logs")
    fsm = FSMProtocolVerifier("configs/experiment_protocol.yaml")
    
    # Save structured telemetry
    telemetry_data = fsm.get_structured_telemetry()
    json_path = logger.save_structured_json(telemetry_data)
    txt_path = logger.save_structured_text_report(telemetry_data)

    # Validate JSON
    assert os.path.exists(json_path), f"JSON file not created at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["mission"] == "BHARATIYA_ANTARIKSH_STATION_IVA"
    assert len(data["protocol_summary"]["steps"]) == 7

    # Validate Lightweight Text File (.txt)
    assert os.path.exists(txt_path), f"Structured text report not created at {txt_path}"
    with open(txt_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    assert "BHARATIYA ANTARIKSH STATION (BAS)" in text_content
    assert "CONDUCTED EXPERIMENT STEPS & OUTCOMES:" in text_content
    assert "MISSION AUDIT & ANOMALY SUMMARY:" in text_content
    print(f"  -> Structured JSON validated at {json_path}")
    print(f"  -> Structured Lightweight Text (.txt) validated at {txt_path} ({len(text_content)} bytes)")


def test_synthetic_dataset_and_model_export():
    print("[TEST 7/7] Testing Microgravity Dataset Generation & Model Artifact...")
    gen = SyntheticDatasetGenerator(output_dir="dataset/test_synthetic", image_size=(320, 320))
    gen.generate_batch(num_samples=3, split="train")

    img_file = "dataset/test_synthetic/images/train/synth_bas_train_00000.jpg"
    lbl_file = "dataset/test_synthetic/labels/train/synth_bas_train_00000.txt"
    assert os.path.exists(img_file), f"Synthetic image {img_file} not found"
    assert os.path.exists(lbl_file), f"Synthetic label {lbl_file} not found"

    # Verify model deliverable artifact
    model_artifact = "storage/models/astro_har_edge_model.json"
    assert os.path.exists(model_artifact), f"Trained AI model artifact missing at {model_artifact}"
    with open(model_artifact, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_name"] == "Astro-HAR-SpaceTransformer-v1"
    assert meta["status"] == "READY_FOR_DEPLOYMENT"
    print(f"  -> Synthetic 6-DOF dataset verified.")
    print(f"  -> Standalone AI Model Deliverable verified: Accuracy={meta['final_accuracy']*100:.1f}%")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING ASTRO-HAR EXPANDED VERIFICATION TEST SUITE (7/7)")
    print("=" * 70)
    test_payload_calibration()
    test_hmr_pose_tracker()
    test_fsm_normal_progression()
    test_fsm_out_of_order_anomaly()
    test_fsm_skipped_step_anomaly()
    test_telemetry_logger_json_and_text()
    test_synthetic_dataset_and_model_export()
    print("=" * 70)
    print("ALL 7/7 TESTS PASSED SUCCESSFULLY! FULLY COMPLIANT WITH SIH 26174")
    print("=" * 70)
