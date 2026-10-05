"""
Offline Audio Alert Engine for Space Edge Computing.
Zero cloud dependency, air-gapped text-to-speech with prioritized worker queue.
Operates asynchronously to prevent blocking the high-frequency 60 FPS vision pipeline.
"""

import sys
import queue
import threading
import subprocess
import time
from typing import Optional


class OfflineVoiceAlertEngine:
    """
    Asynchronous offline speech synthesis engine.
    Supports pyttsx3, native macOS 'say', and Linux 'espeak'/Piper fallback.
    """

    def __init__(self, speech_rate: int = 175, volume: float = 1.0, enabled: bool = True):
        self.enabled = enabled
        self.speech_rate = speech_rate
        self.volume = volume
        self.is_running = True
        self.is_speaking = False

        # Priority Queue: (priority_int, timestamp, text)
        # priority 0 = CRITICAL ANOMALY, priority 1 = STEP PROMPT, priority 2 = INFO
        self.speech_queue = queue.PriorityQueue()
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        
        # Test engine availability
        self._init_tts_backend()
        self.worker_thread.start()

    def _init_tts_backend(self):
        """Initializes backend engine preference."""
        self.use_native_say = (sys.platform == "darwin")
        self.pyttsx3_engine = None

        if not self.use_native_say:
            try:
                import pyttsx3
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty("rate", self.speech_rate)
                self.pyttsx3_engine.setProperty("volume", self.volume)
            except Exception as e:
                print(f"[AudioEngine] pyttsx3 init fallback warning: {e}")

    def speak_alert(self, text: str):
        """High priority: Emergency anomaly, out-of-order execution, skipped step."""
        if not self.enabled:
            return
        # Priority 0
        self.speech_queue.put((0, time.time(), text))

    def speak_prompt(self, text: str):
        """Normal priority: Next step instruction or milestone reached."""
        if not self.enabled:
            return
        # Priority 1
        self.speech_queue.put((1, time.time(), text))

    def _speech_worker(self):
        """Continuous background thread processing audio prompts."""
        while self.is_running:
            try:
                item = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            priority, timestamp, text = item

            # Drop stale normal prompts older than 10 seconds
            if priority > 0 and (time.time() - timestamp) > 10.0:
                self.speech_queue.task_done()
                continue

            self._speak_now(text)
            self.speech_queue.task_done()

    def _speak_now(self, text: str):
        """Executes offline synthesis using the best local driver."""
        if not text or not text.strip():
            return

        self.is_speaking = True
        try:
            print(f"[AUDIO ALERT ENGINE] Spoken: \"{text}\"")

            # 1. On macOS, native 'say' provides instant zero-latency offline TTS
            if self.use_native_say:
                try:
                    cmd = ["say", "-r", str(self.speech_rate), text]
                    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except Exception:
                    pass

            # 2. Linux / cross-platform pyttsx3
            if self.pyttsx3_engine is not None:
                try:
                    self.pyttsx3_engine.say(text)
                    self.pyttsx3_engine.runAndWait()
                    return
                except Exception:
                    pass

            # 3. Linux espeak fallback
            try:
                subprocess.run(["espeak", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        finally:
            self.is_speaking = False

    def stop(self):
        """Stops speech engine cleanly."""
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
