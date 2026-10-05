// Astro-HAR Mission Control Dashboard WebSocket & Telemetry Client
let socket = null;
let reconnectTimer = null;
let missionStartTime = Date.now();

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initWebSocket();
  initControlButtons();
});

function initClock() {
  setInterval(() => {
    const elapsed = Math.floor((Date.now() - missionStartTime) / 1000);
    const hrs = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    const metEl = document.getElementById("met-clock");
    if (metEl) metEl.innerText = `MET ${hrs}:${mins}:${secs}`;
  }, 1000);
}

function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log("[Astro-HAR] Telemetry WebSocket connected.");
    const indicator = document.getElementById("status-indicator");
    if (indicator) {
      indicator.innerHTML = `<span class="pulse-dot"></span> EDGE LINK ACTIVE`;
      indicator.style.color = "var(--accent-green)";
    }
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateTelemetryDashboard(data);
    } catch (e) {
      console.error("[Astro-HAR] Error parsing telemetry:", e);
    }
  };

  socket.onclose = () => {
    console.warn("[Astro-HAR] WebSocket disconnected. Reconnecting in 2s...");
    const indicator = document.getElementById("status-indicator");
    if (indicator) {
      indicator.innerHTML = `<span class="pulse-dot" style="background:var(--accent-amber)"></span> RECONNECTING...`;
      indicator.style.color = "var(--accent-amber)";
    }
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(initWebSocket, 2000);
  };
}

function updateTelemetryDashboard(data) {
  // 1. Current Step Info
  const currStep = data.current_step;
  const stepNumEl = document.getElementById("step-num-display");
  const stepNameEl = document.getElementById("step-name-display");
  const stepDescEl = document.getElementById("step-desc-display");
  const stepTargetEl = document.getElementById("step-target-display");
  const statusTag = document.getElementById("step-status-tag");

  if (currStep) {
    if (stepNumEl) stepNumEl.innerText = `STEP ${currStep.id} OF ${data.total_steps || 7}`;
    if (stepNameEl) stepNameEl.innerText = currStep.name;
    if (stepDescEl) stepDescEl.innerText = currStep.description;
    if (stepTargetEl) stepTargetEl.innerText = (currStep.target_object || "").toUpperCase();
    
    if (statusTag) {
      statusTag.innerText = currStep.status;
      statusTag.className = "step-status-tag " + (currStep.status === "COMPLETED" ? "tag-completed" : "tag-inprogress");
    }
  } else if (data.is_completed) {
    if (stepNumEl) stepNumEl.innerText = "EXPERIMENT COMPLETE";
    if (stepNameEl) stepNameEl.innerText = "All Protocols Verified";
    if (stepDescEl) stepDescEl.innerText = "Scientific experiment execution verified without protocol violations.";
    if (stepTargetEl) stepTargetEl.innerText = "ALL CLEAR";
    if (statusTag) {
      statusTag.innerText = "VERIFIED";
      statusTag.className = "step-status-tag tag-completed";
    }
  }

  // 2. Proactive Next Step Suggestion (SIH 26174 Requirement)
  const nextStep = data.next_suggested_step;
  const nextCard = document.getElementById("next-step-card");
  const nextCodeEl = document.getElementById("next-step-code");
  const nextTitleEl = document.getElementById("next-step-title");
  const nextPromptEl = document.getElementById("next-step-prompt");

  if (nextStep && nextCard) {
    nextCard.style.display = "block";
    if (nextCodeEl) nextCodeEl.innerText = `STEP ${nextStep.id}`;
    if (nextTitleEl) nextTitleEl.innerText = `Next: ${nextStep.name}`;
    if (nextPromptEl) {
      nextPromptEl.innerText = `👉 Expected Target: ${nextStep.target_object.toUpperCase()} | Action: ${nextStep.expected_action}`;
    }
  } else if (data.is_completed && nextCard) {
    if (nextCodeEl) nextCodeEl.innerText = "FINAL";
    if (nextTitleEl) nextTitleEl.innerText = "Next: Stowage & Telemetry Downlink";
    if (nextPromptEl) nextPromptEl.innerText = "👉 All 7 experiment steps completed. Secure rack and transmit data.";
  }

  // 3. Voice Prompt text & Audio Waveform animation
  const voiceEl = document.getElementById("voice-prompt-text");
  if (voiceEl && data.active_voice_prompt) {
    voiceEl.innerText = `"${data.active_voice_prompt}"`;
  }
  const waveformBox = document.getElementById("audio-waveform-indicator");
  const waveStatus = document.getElementById("waveform-status-text");
  if (waveformBox && waveStatus) {
    if (data.is_speaking) {
      waveformBox.classList.add("active");
      waveStatus.innerText = "● AI COPILOT SPEAKING";
    } else {
      waveformBox.classList.remove("active");
      waveStatus.innerText = "OFFLINE TTS READY";
    }
  }

  // 4. Progress Bar
  const progressPct = data.completion_percentage || 0;
  const pctEl = document.getElementById("progress-percent-val");
  const barEl = document.getElementById("progress-fill-bar");
  if (pctEl) pctEl.innerText = `${progressPct}%`;
  if (barEl) barEl.style.width = `${progressPct}%`;

  // 5. Anomaly Alert Banner
  const alertBanner = document.getElementById("anomaly-alert-banner");
  if (alertBanner) {
    if (data.active_alert) {
      alertBanner.classList.add("active");
      const titleEl = document.getElementById("alert-banner-title");
      const descEl = document.getElementById("alert-banner-desc");
      if (titleEl) titleEl.innerText = data.active_alert.type || "ANOMALY DETECTED";
      if (descEl) descEl.innerText = data.active_alert.message || "Out-of-order execution detected.";
    } else {
      alertBanner.classList.remove("active");
    }
  }

  // 6. Telemetry Gauges & Hardware Profiler
  if (data.edge_profiler) {
    const profEl = document.getElementById("prof-metric");
    if (profEl) {
      profEl.innerText = `${data.edge_profiler.latency_ms}ms / ${data.edge_profiler.fps} FPS`;
    }
  }

  if (data.spatial_context) {
    const ctx = data.spatial_context;
    const hoiEl = document.getElementById("gauge-hoi-target");
    const distEl = document.getElementById("gauge-distance");
    if (hoiEl) hoiEl.innerText = (ctx.hand_contact_target || "NONE").toUpperCase();
    if (distEl) distEl.innerText = ctx.min_dist_norm !== undefined ? ctx.min_dist_norm.toFixed(2) : "--";

    // 3D HMR Orientation-Agnostic Posture
    const hmr = ctx.hmr_pose;
    const hmrEl = document.getElementById("gauge-hmr-orientation");
    if (hmr && hmrEl) {
      hmrEl.innerText = `ROLL: ${hmr.roll_deg}° | PITCH: ${hmr.pitch_deg}° | 3D MESH: ${Math.round((hmr.mesh_confidence || 0.9) * 100)}%`;
    }
  }
  const actEl = document.getElementById("gauge-action");
  if (actEl && data.predicted_action) {
    actEl.innerText = data.predicted_action.toUpperCase();
  }
  const confEl = document.getElementById("gauge-confidence");
  if (confEl && data.action_confidence !== undefined) {
    confEl.innerText = `${Math.round(data.action_confidence * 100)}%`;
  }

  // 7. Checklist update
  if (data.steps && data.steps.length > 0) {
    updateChecklist(data.steps, data.current_step ? data.current_step.id : -1);
  }

  // 8. Event Log Stream
  if (data.latest_event) {
    appendEventLog(data.latest_event);
  }
}

function updateChecklist(steps, currentStepId) {
  const container = document.getElementById("steps-checklist");
  if (!container) return;

  container.innerHTML = "";
  steps.forEach(step => {
    const row = document.createElement("div");
    const isActive = step.id === currentStepId;
    const isCompleted = step.status === "COMPLETED";

    row.className = `step-row ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`;
    row.innerHTML = `
      <div class="step-left">
        <div class="step-check-icon">${isCompleted ? '✓' : step.id}</div>
        <div class="step-title-text">${step.name}</div>
      </div>
      <div class="step-time-text">${step.duration_sec > 0 ? step.duration_sec + 's' : (isActive ? 'ACTIVE' : '--')}</div>
    `;
    container.appendChild(row);
  });
}

function appendEventLog(eventText) {
  const logBox = document.getElementById("event-log-container");
  if (!logBox) return;

  const entry = document.createElement("div");
  entry.className = "log-entry";
  const nowStr = new Date().toLocaleTimeString();
  entry.innerHTML = `<span class="log-time">[${nowStr}]</span> ${eventText}`;
  logBox.appendChild(entry);
  logBox.scrollTop = logBox.scrollHeight;
}

function initControlButtons() {
  // 1. Inject Out-of-Sequence Anomaly
  const btnOutOfSequence = document.getElementById("btn-inject-out-of-sequence") || document.getElementById("btn-inject-anomaly");
  if (btnOutOfSequence) {
    btnOutOfSequence.addEventListener("click", async () => {
      btnOutOfSequence.disabled = true;
      btnOutOfSequence.innerText = "Triggering...";
      try {
        await fetch("/api/control/inject_out_of_sequence", { method: "POST" });
        appendEventLog(`<span class="log-alert">⚠️ ANOMALY: Out-of-sequence step action injected!</span>`);
      } catch (e) {
        console.error(e);
      } finally {
        setTimeout(() => {
          btnOutOfSequence.disabled = false;
          btnOutOfSequence.innerHTML = `⚠️ Out-of-Sequence`;
        }, 1500);
      }
    });
  }

  // 2. Inject Skipped Step Anomaly
  const btnSkipStep = document.getElementById("btn-inject-skip-step");
  if (btnSkipStep) {
    btnSkipStep.addEventListener("click", async () => {
      btnSkipStep.disabled = true;
      btnSkipStep.innerText = "Triggering...";
      try {
        await fetch("/api/control/inject_skipped_step", { method: "POST" });
        appendEventLog(`<span class="log-alert">⚠️ ANOMALY: Protocol step skip injected!</span>`);
      } catch (e) {
        console.error(e);
      } finally {
        setTimeout(() => {
          btnSkipStep.disabled = false;
          btnSkipStep.innerHTML = `⏭️ Skip Step`;
        }, 1500);
      }
    });
  }

  // 3. Next Step Override
  const btnNext = document.getElementById("btn-force-next");
  if (btnNext) {
    btnNext.addEventListener("click", async () => {
      await fetch("/api/control/next_step", { method: "POST" });
      appendEventLog("Manual override: Step advanced.");
    });
  }

  // 4. Reset Button
  const btnReset = document.getElementById("btn-reset-protocol");
  if (btnReset) {
    btnReset.addEventListener("click", async () => {
      await fetch("/api/control/reset", { method: "POST" });
      missionStartTime = Date.now();
      appendEventLog("Experiment Protocol Reset.");
    });
  }

  // 5. View Text Report Modal
  const btnViewText = document.getElementById("btn-view-text-report");
  const modalEl = document.getElementById("telemetry-text-modal");
  const modalClose = document.getElementById("btn-close-modal");
  const modalCopy = document.getElementById("btn-copy-text-report");
  const modalContent = document.getElementById("telemetry-text-content");

  if (btnViewText && modalEl) {
    btnViewText.addEventListener("click", async () => {
      try {
        modalEl.style.display = "flex";
        if (modalContent) modalContent.innerText = "Fetching live structured telemetry report...";
        const res = await fetch("/api/telemetry/text_content");
        const json = await res.json();
        if (modalContent) modalContent.innerText = json.content || "No telemetry records found.";
        appendEventLog("Opened on-board telemetry text report viewer.");
      } catch (e) {
        if (modalContent) modalContent.innerText = "Error loading telemetry content.";
      }
    });
  }

  if (modalClose && modalEl) {
    modalClose.addEventListener("click", () => {
      modalEl.style.display = "none";
    });
  }

  if (modalCopy && modalContent) {
    modalCopy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(modalContent.innerText);
        modalCopy.innerText = "✓ Copied!";
        setTimeout(() => { modalCopy.innerText = "📋 Copy"; }, 1500);
      } catch (e) {
        console.error(e);
      }
    });
  }

  // 6. Download Text Report (.txt)
  const btnDownloadText = document.getElementById("btn-download-text");
  if (btnDownloadText) {
    btnDownloadText.addEventListener("click", () => {
      window.location.href = "/api/telemetry/download_text";
      appendEventLog("Lightweight text summary (.txt) downloaded.");
    });
  }

  // 7. Download JSON Telemetry
  const btnDownloadJson = document.getElementById("btn-download-telemetry");
  if (btnDownloadJson) {
    btnDownloadJson.addEventListener("click", () => {
      window.location.href = "/api/telemetry/download";
      appendEventLog("JSON telemetry downloaded.");
    });
  }

  // 8. Download Session MP4 Video
  const btnDownloadVideo = document.getElementById("btn-download-video");
  if (btnDownloadVideo) {
    btnDownloadVideo.addEventListener("click", () => {
      window.location.href = "/api/video/download";
      appendEventLog("Local session video (.mp4) downloaded.");
    });
  }

  // 9. Toggle Audio
  const btnAudio = document.getElementById("btn-toggle-audio");
  if (btnAudio) {
    btnAudio.addEventListener("click", async () => {
      const res = await fetch("/api/control/toggle_audio", { method: "POST" });
      const data = await res.json();
      btnAudio.innerHTML = data.audio_enabled ? "🔊 Voice: ON" : "🔇 Voice: MUTED";
      appendEventLog(`Audio voice engine toggled: ${data.audio_enabled ? 'ON' : 'MUTED'}`);
    });
  }

  // 10. Multi-Protocol Switcher Dropdown (Modular SOP Support)
  const protoSelect = document.getElementById("protocol-selector");
  if (protoSelect) {
    protoSelect.addEventListener("change", async (e) => {
      const chosen = e.target.value;
      try {
        const res = await fetch("/api/control/switch_protocol", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ protocol: chosen })
        });
        const data = await res.json();
        if (data.status === "SUCCESS") {
          missionStartTime = Date.now();
          appendEventLog(`Switched Experiment Protocol to: ${data.protocol_title}`);
        }
      } catch (err) {
        console.error("Failed to switch protocol:", err);
      }
    });
  }
}
