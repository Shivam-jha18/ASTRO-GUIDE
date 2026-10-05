"""
Interactive Dataset Recorder with Live Step-Tagging.
Allows teams to record webcam or desk experiments with real red & yellow boxes
and automatically tag the step sequences for custom dataset training.
"""

import os
import cv2
import time
import argparse


def run_recorder(output_dir: str = "dataset/recorded_sessions", camera_idx: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[Recorder Error] Cannot open camera index {camera_idx}")
        return

    print("=" * 60)
    print("ASTRO-HAR DATASET RECORDER (SIH 26174)")
    print("Keys:")
    print("  [0-7] : Tag active Step ID (0: Idle, 1: Open Main, 2: Red Box, 3: Inspect Red, 4: Yellow Box, 5: Interact, 6: Return, 7: Close)")
    print("  [R]   : Start/Stop Recording")
    print("  [S]   : Capture Single Snapshot")
    print("  [Q]   : Quit")
    print("=" * 60)

    current_tag = "step_0_idle"
    is_recording = False
    writer = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = frame.shape[:2]

        # Overlay recording and tag info
        status_text = f"REC: {'ACTIVE' if is_recording else 'STANDBY'} | TAG: {current_tag.upper()} | Frames: {frame_count}"
        color = (0, 0, 255) if is_recording else (0, 255, 0)
        cv2.putText(display, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("Astro-HAR Dataset Recorder", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            if not is_recording:
                # Start recording
                t_str = time.strftime("%Y%m%d_%H%M%S")
                rec_path = os.path.join(output_dir, f"{t_str}_{current_tag}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(rec_path, fourcc, 30.0, (w, h))
                is_recording = True
                frame_count = 0
                print(f"[Recorder] Started: {rec_path}")
            else:
                is_recording = False
                if writer:
                    writer.release()
                    writer = None
                print(f"[Recorder] Stopped. Total frames saved: {frame_count}")
        elif ord('0') <= key <= ord('7'):
            step_num = key - ord('0')
            tags = [
                "step_0_idle",
                "step_1_open_main_box",
                "step_2_extract_red_box",
                "step_3_inspect_red_box",
                "step_4_extract_yellow_box",
                "step_5_execute_interaction",
                "step_6_return_boxes",
                "step_7_close_main_box"
            ]
            current_tag = tags[step_num]
            print(f"[Recorder] Switched tag: {current_tag}")

        if is_recording and writer:
            writer.write(frame)
            frame_count += 1

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=str, default="dataset/recorded_sessions")
    args = parser.parse_args()
    run_recorder(output_dir=args.output, camera_idx=args.camera)
