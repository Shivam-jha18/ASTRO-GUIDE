"""
Timestamped Structured Telemetry Log Generator.
Outputs lightweight, standardized JSON and CSV telemetry logs for mission archive
and ground-station downlink as required by SIH Problem 26174.
"""

import os
import json
import csv
import time
from typing import Dict, Any, List, Optional


class TelemetryLogger:
    """
    Manages structured, lightweight telemetry generation for scientific experiments.
    """

    def __init__(self, output_dir: str = "telemetry_data/logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.session_id = time.strftime("BAS_EXP_%Y%m%d_%H%M%S")
        self.current_json_path = os.path.join(self.output_dir, f"{self.session_id}_telemetry.json")
        self.current_csv_path = os.path.join(self.output_dir, f"{self.session_id}_events.csv")
        self.current_txt_path = os.path.join(self.output_dir, f"{self.session_id}_summary.txt")

        self.events: List[Dict[str, Any]] = []
        self._init_csv()
        self._init_txt()

    def _init_csv(self):
        """Initializes CSV header."""
        with open(self.current_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp_ISO",
                "Unix_Time",
                "Event_Type",
                "Step_ID",
                "Step_Name",
                "Outcome_Status",
                "Duration_Sec",
                "Anomaly_Details"
            ])

    def _init_txt(self):
        """Initializes the structured lightweight text summary log."""
        now_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        header = [
            "=" * 78,
            "BHARATIYA ANTARIKSH STATION (BAS) - IVA SCIENTIFIC EXPERIMENT LOG",
            f"Session ID:       {self.session_id}",
            f"Timestamp (UTC):  {now_utc}",
            "Spacecraft Unit:  EDGE_PAYLOAD_AI_01",
            "Mode:             AIR-GAPPED STANDALONE EDGE AI (SIH-2026 / PROBLEM 26174)",
            "=" * 78,
            "\n[EXPERIMENT INITIALIZED - AWAITING CREW PROTOCOL EXECUTION]\n"
        ]
        with open(self.current_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")

    def log_event(
        self,
        event_type: str,
        step_id: int,
        step_name: str,
        status: str,
        duration_sec: float = 0.0,
        anomaly_details: str = "None"
    ):
        """Appends an event row to the live CSV and event buffer."""
        now = time.time()
        iso_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

        row = {
            "timestamp_iso": iso_str,
            "unix_time": round(now, 3),
            "event_type": event_type,
            "step_id": step_id,
            "step_name": step_name,
            "status": status,
            "duration_sec": duration_sec,
            "anomaly_details": anomaly_details
        }
        self.events.append(row)

        with open(self.current_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                iso_str,
                round(now, 3),
                event_type,
                step_id,
                step_name,
                status,
                duration_sec,
                anomaly_details
            ])

    def save_structured_json(self, fsm_telemetry: Dict[str, Any]) -> str:
        """
        Dumps complete, structured lightweight JSON telemetry report.
        """
        payload = {
            "metadata": {
                "mission": "BHARATIYA_ANTARIKSH_STATION_IVA",
                "spacecraft_unit": "EDGE_PAYLOAD_AI_01",
                "session_id": self.session_id,
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "format_version": "1.0-SIH26174",
            },
            "protocol_summary": fsm_telemetry,
            "event_stream": self.events
        }

        with open(self.current_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Simultaneously save structured lightweight text report
        self.save_structured_text_report(fsm_telemetry)

        return self.current_json_path

    def save_structured_text_report(self, fsm_telemetry: Dict[str, Any]) -> str:
        """
        Generates a timestamped, structured, lightweight human-and-ground readable
        text report (.txt) of conducted experiment steps and outcomes/status.
        Specifically fulfills SIH Problem 26174 lightweight text file requirement.
        """
        now_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        protocol_title = fsm_telemetry.get("protocol_title", "Microgravity Scientific Experiment")
        protocol_code = fsm_telemetry.get("protocol_code", "EXP_26174")
        is_completed = fsm_telemetry.get("is_completed", False)
        total_duration = fsm_telemetry.get("total_duration_sec", 0.0)
        completion_pct = fsm_telemetry.get("completion_percentage", 0)
        steps = fsm_telemetry.get("steps", [])

        lines = [
            "=" * 78,
            "BHARATIYA ANTARIKSH STATION (BAS) - ON-BOARD EXPERIMENT VERIFICATION REPORT",
            "=" * 78,
            f"Mission / Payload:     {fsm_telemetry.get('mission', 'BAS_IVA_EXPERIMENT')}",
            f"Session ID:            {self.session_id}",
            f"Report Generated (UTC):{now_utc}",
            f"Hardware Platform:     EDGE_PAYLOAD_AI_01 (Offline Standalone Operation)",
            f"Protocol Code:         {protocol_code}",
            f"Protocol Title:        {protocol_title}",
            f"Completion Status:     {'VERIFIED COMPLETED' if is_completed else 'IN PROGRESS / PARTIAL'}",
            f"Overall Progress:      {completion_pct}%",
            f"Total Elapsed Time:    {total_duration} seconds",
            "-" * 78,
            "CONDUCTED EXPERIMENT STEPS & OUTCOMES:",
            "-" * 78,
            f"{'ID':<4} {'STEP NAME':<34} {'TARGET':<12} {'DURATION':<10} {'STATUS':<12}"
        ]
        lines.append("-" * 78)

        total_anomalies = 0
        for s in steps:
            s_id = s.get("id", 0)
            name = s.get("name", "Unknown Step")
            target = s.get("target_object", "N/A")
            duration = f"{s.get('duration_sec', 0.0):.1f}s"
            status = s.get("status", "PENDING")
            conf = s.get("confidence", 0.0)
            anomalies = s.get("anomalies", [])
            total_anomalies += len(anomalies)

            status_str = f"[{status}]"
            lines.append(f"{s_id:<4} {name[:33]:<34} {target[:11]:<12} {duration:<10} {status_str:<12}")
            if conf > 0.0:
                lines.append(f"     -> Action: {s.get('expected_action', 'N/A')} | AI Confidence: {conf * 100:.1f}%")
            if anomalies:
                for a in anomalies:
                    lines.append(f"     -> ⚠️ ANOMALY: [{a.get('type', 'VIOLATION')}] {a.get('message', 'Out of order action')}")

        lines.append("-" * 78)
        lines.append("MISSION AUDIT & ANOMALY SUMMARY:")
        lines.append(f"Total Anomalies Logged: {total_anomalies}")
        if total_anomalies == 0:
            lines.append("Protocol Compliance:   100% PERFECT ADHERENCE (Zero sequence violations)")
        else:
            lines.append(f"Protocol Compliance:   {total_anomalies} ANOMALY EVENT(S) DETECTED AND RESOLVED")

        lines.append("-" * 78)
        lines.append("RECENT TELEMETRY EVENTS STREAM:")
        for ev in self.events[-10:]:
            lines.append(f"  {ev.get('timestamp_iso')} | {ev.get('event_type'):<16} | Step {ev.get('step_id')}: {ev.get('status')} | {ev.get('anomaly_details')}")

        lines.append("=" * 78)
        lines.append("END OF STRUCTURED TELEMETRY REPORT [ARCHIVED LOCALLY]")
        lines.append("=" * 78)

        report_content = "\n".join(lines) + "\n"
        with open(self.current_txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return self.current_txt_path

    def get_latest_json_content(self) -> str:
        """Reads back current JSON log as a string for GUI display or API export."""
        if os.path.exists(self.current_json_path):
            with open(self.current_json_path, "r", encoding="utf-8") as f:
                return f.read()
        return "{}"

    def get_latest_text_content(self) -> str:
        """Reads back current structured text summary report."""
        if os.path.exists(self.current_txt_path):
            with open(self.current_txt_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""
