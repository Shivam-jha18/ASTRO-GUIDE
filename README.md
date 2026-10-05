# 🛰️ Astro-HAR: Autonomous Edge AI Copilot for Astronaut Experiment Verification
### Smart India Hackathon 2026 | Problem Statement 26174
**Target Deployment:** Bharatiya Antariksh Station (BAS) & Deep-Space Lunar Missions (IVA Payloads)

---

## 📌 Executive Summary
During deep space exploration (Lunar and Bharatiya Antariksh Station missions), speed-of-light communication latency and constrained uplink bandwidth render real-time ground control intervention impossible. 

**Astro-HAR** is an air-gapped, standalone edge AI assistant deployed directly inside the payload rack. It continuously monitors astronaut experiment procedures using fixed-payload cameras, verifies standard operating protocol (SOP) compliance in real-time, proactively guides astronauts through multi-step procedures, speaks immediate voice alerts upon detecting skipped or out-of-order actions, streams the experiment to station IP, and records timestamped structured JSON telemetry for mission archive.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│               LOCAL EDGE PROCESSING UNIT                │
│                                                         │
│ [ Payload Camera ]                                      │
│         │                                               │
│         ▼                                               │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 1. Frame Acquisition & Pre-processing (60 FPS)      │ │
│ └──────────────────────────┬──────────────────────────┘ │
│                            │                            │
│                            ▼                            │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 2. Orientation-Agnostic Spatial Engine              │ │
│ │    • Payload-Centric Canonical Rack Homography      │ │
│ │    • YOLOv8-Nano Object & Hand-Bounding Detector   │ │
│ └──────────────────────────┬──────────────────────────┘ │
│                            │                            │
│                            ▼                            │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 3. Space-HAR Transformer (Action Classification)    │ │
│ └──────────────────────────┬──────────────────────────┘ │
│                            │                            │
│                            ▼                            │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 4. Dynamic Finite State Machine & Protocol Verifier │ │
│ └───────┬──────────────────────────┬──────────────────┘ │
└─────────┼──────────────────────────┼────────────────────┘
          │                          │
          ▼                          ▼
┌──────────────────────────────────────┐   ┌──────────────────────────────────────┐
│       AUDIO ALERT ENGINE             │   │       ON-BOARD GUI DASHBOARD         │
│  - Offline Native / Pyttsx3 TTS      │   │  - Real-time Video Stream (RTSP/IP)  │
│  - Trigger: Missing / Out-of-order   │   │  - Active Step & Voice Prompts       │
│    step execution                    │   │  - Local MP4 Video Storage & Export  │
│  - Air-gapped, zero cloud call       │   │  - Timestamped JSON Telemetry Log    │
└──────────────────────────────────────┘   └──────────────────────────────────────┘
```

---

## 🧪 Sample Experiment Protocol: Red & Yellow Box (7-Step SOP)

Configured inside [`configs/experiment_protocol.yaml`](file:///Users/jahnvisingh/Desktop/Astro_sih26174/configs/experiment_protocol.yaml):

| Step | Action Code | Name | Target Object | Prerequisites | Voice Prompt / Guidance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `STEP_1_OPEN_MAIN_BOX` | Unlatch & Open Main Box | `main_box` | *None* | *"Protocol started. Step one: Please unlatch and open the main containment box."* |
| **2** | `STEP_2_EXTRACT_RED_BOX` | Extract Red Box | `red_box` | Step 1 | *"Main box opened. Step two: Please retrieve the Red specimen box."* |
| **3** | `STEP_3_INSPECT_RED_BOX` | Inspect / Open Red Box | `red_box` | Step 2 | *"Red box extracted. Step three: Inspect the specimen inside the Red box."* |
| **4** | `STEP_4_EXTRACT_YELLOW_BOX`| Extract Yellow Box | `yellow_box` | Step 3 | *"Inspection verified. Step four: Please retrieve the Yellow container."* |
| **5** | `STEP_5_EXECUTE_INTERACTION`| Perform Payload Slot Interaction | `payload_slot` | Step 4 | *"Both containers ready. Step five: Execute sample interaction or slot insertion."* |
| **6** | `STEP_6_RETURN_BOXES` | Return Containers to Main Box | `main_box` | Step 5 | *"Interaction complete. Step six: Return both Red and Yellow boxes into the main container."* |
| **7** | `STEP_7_CLOSE_MAIN_BOX` | Close & Latch Main Box | `main_box` | Step 6 | *"Specimens stowed. Step seven: Close and lock the main containment box."* |

### Anomaly & Voice Alert Verification
- If an astronaut attempts to extract the **Yellow Box** before inspecting the **Red Box**, the FSM immediately fires an **`OUT_OF_SEQUENCE`** alert:
  > 🎙️ *"Alert! Yellow box extracted out of sequence. Red box procedure incomplete."*
- If an astronaut attempts to close the container with loose samples outside:
  > 🎙️ *"Safety violation! Loose items detected. Secure Red and Yellow boxes inside main unit."*

---

## 🚀 Quick Start Guide

### 1. Environment Setup
The project uses a Python 3.12 virtual environment:
```bash
# If venv is not already activated:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launching the End-to-End System

#### Option A: Running with Built-in 0-G Simulator (Zero Hardware Required)
Demonstrates the full 7-step sequence and anomaly detection without needing physical boxes:
```bash
./run.py --synthetic
```

#### Option B: Running with Physical Payload Camera / Webcam
Connect any USB/UVC camera or use your built-in Mac webcam:
```bash
./run.py --camera 0
```

### 3. Open the Mission Control Dashboard
Navigate to:
- **Cockpit HUD:** `http://localhost:8080`
- **Network Video Stream over IP:** `http://<YOUR_IP>:8080/stream/video_feed`

---

## 📂 Project Directory Structure

```
Astro_sih26174/
├── configs/
│   ├── experiment_protocol.yaml       # 7-Step SOP definitions, audio alerts, and timeouts
│   └── pipeline_config.yaml           # Edge compute, camera, stream IP, and model settings
├── core/
│   ├── acquisition.py                 # Threaded 60 FPS frame grabber & 0-G synthetic feed
│   ├── payload_calibration.py         # Canonical rack homography (orientation-agnostic frame)
│   ├── hmr_pose_tracker.py            # Orientation-Agnostic 3D Human Mesh Recovery (HMR)
│   ├── spatial_engine.py              # Object detector, astronaut hand tracker & HOI analyzer
│   ├── action_transformer.py          # Space-HAR Transformer (spatio-temporal action classifier)
│   └── fsm_verifier.py                # Dynamic FSM Protocol Verifier (Skipped & Out-of-Order alerts)
├── audio/
│   └── alert_engine.py                # Asynchronous air-gapped offline voice alert engine
├── storage/
│   ├── video_recorder.py              # Threaded continuous MP4 session recorder
│   ├── telemetry_logger.py            # Structured lightweight .txt, JSON & CSV telemetry generator
│   └── models/
│       └── astro_har_edge_model.json  # Trained standalone edge AI model deliverable
├── dashboard/
│   ├── server.py                      # FastAPI server streaming video & WebSocket telemetry
│   ├── templates/index.html           # Bharatiya Antariksh Station Mission Control HUD
│   └── static/
│       ├── css/style.css              # Dark-mode glassmorphic mission control styling
│       └── js/dashboard.js            # Real-time WebSocket telemetry client & control triggers
├── dataset_generator/
│   ├── synthetic_box_generator.py     # Generates annotated YOLO microgravity dataset (6-DOF)
│   ├── train_offline_model.py         # Offline standalone model training & validation pipeline
│   ├── microgravity_augmentor.py      # Photometric & 360-degree rotation augmentations
│   └── dataset_recorder.py            # Interactive webcam recorder with keyboard step-tagging
├── tests/
│   └── test_pipeline.py               # Comprehensive unit & integration test suite (7/7 passing)
├── main.py                            # Central edge pipeline orchestrator
├── run.py                             # Executable unified launcher (--stream-ip, --synthetic, --camera)
└── requirements.txt                   # Tested edge dependencies
```

---

## 🛠️ Synthetic Dataset Generation

To generate custom synthetic datasets for training YOLOv8-nano on 6-DOF microgravity orientations:
```bash
# Generates 500 annotated training images with random microgravity rotations & YOLO labels:
./venv/bin/python -c "from dataset_generator import SyntheticDatasetGenerator; SyntheticDatasetGenerator().generate_batch(num_samples=500)"
```

To record your own desk setup with physical red and yellow boxes:
```bash
./venv/bin/python dataset_generator/dataset_recorder.py --camera 0
```
- Press keys `1`–`7` to tag the active step.
- Press `R` to start/stop saving MP4 session clips.

---

## 📊 Structured Lightweight JSON Telemetry

The system continuously logs flight telemetry to [`telemetry_data/logs/`](file:///Users/jahnvisingh/Desktop/Astro_sih26174/telemetry_data/logs/):
```json
{
  "metadata": {
    "mission": "BHARATIYA_ANTARIKSH_STATION_IVA",
    "spacecraft_unit": "EDGE_PAYLOAD_AI_01",
    "session_id": "BAS_EXP_20260926_192354",
    "format_version": "1.0-SIH26174"
  },
  "protocol_summary": {
    "protocol_code": "ISRO_BAS_EXP_26174",
    "is_completed": true,
    "total_duration_sec": 42.15,
    "completion_percentage": 100,
    "steps": [
      {
        "id": 1,
        "code": "STEP_1_OPEN_MAIN_BOX",
        "name": "Unlatch & Open Main Containment Box",
        "status": "COMPLETED",
        "duration_sec": 3.82,
        "anomalies": []
      },
      {
        "id": 2,
        "code": "STEP_2_EXTRACT_RED_BOX",
        "status": "COMPLETED",
        "duration_sec": 4.12,
        "anomalies": []
      }
    ]
  }
}
```

---

## 🏆 Key SIH 26174 Deliverables Completed

- [x] **Standalone Offline Operation**: 100% air-gapped, zero cloud or internet calls.
- [x] **Orientation-Agnostic Tracking**: Canonical payload-rack homography eliminates gravity-vector dependency.
- [x] **Synthetic Dataset Generation**: Integrated 6-DOF synthetic generator & augmentor.
- [x] **Continuous Video Processing**: Multi-threaded 60 FPS pipeline decoupling camera capture from neural inference.
- [x] **Next Step Guidance**: Contextual voice prompts and HUD card suggestions.
- [x] **Voice-Based Out-of-Order Alerting**: Offline async TTS (`pyttsx3`/native) catches skipped/out-of-sequence steps.
- [x] **Structured Lightweight Telemetry**: Standardized JSON and CSV logs generated on the fly.
- [x] **Stream over IP & Local MP4 Storage**: Continuous background MP4 recorder + MJPEG/RTSP stream over IP.
- [x] **Graphical User Interface**: Futuristic ISRO/BAS dark-mode telemetry console with live interactive controls.
