"""
scripts/interval_labeling_tool.py
---------------------------------
Human Ground-Truth Video Interval Labeling Tool.

Features:
- Web-based frame-accurate video player with sub-frame seek and speed controls.
- Single-key hotkeys for every class in the taxonomy:
    [W] active_work   [T] writing   [R] reading      [D] discussing
    [I] idle          [K] walking   [P] phone        [C] call
    [A] away          [U] uncertain
- Interval marking:
    [ [ ] or [S] : Mark Interval Start (t_start)
    [ ] ] or [E] : Mark Interval End (t_end)
    [Space]      : Play / Pause
    [ , ] / [ . ]: Step 1 Frame Backward / Forward (1/30s)
    [Enter]      : Save Interval
- Annotator Name field persisted across sessions.
- Student / Track selector (Aagam, Swayam, Harshil, Soham, or custom).
- Export to clean CSV in native source-video presentation seconds (t = frame / fps).
- CLI fallback mode available via --cli.
"""

import os
import sys
import glob
import json
import csv
import argparse
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

PORT = 5004
GROUND_TRUTH_DIR = "ground_truth"

CLASSES = [
    {"key": "w", "id": "active_work", "name": "Active Work", "color": "#10b981"},
    {"key": "t", "id": "writing", "name": "Writing", "color": "#14b8a6"},
    {"key": "r", "id": "reading", "name": "Reading", "color": "#06b6d4"},
    {"key": "d", "id": "discussing", "name": "Discussing", "color": "#38bdf8"},
    {"key": "i", "id": "idle", "name": "Idle / Distracted", "color": "#eab308"},
    {"key": "k", "id": "walking", "name": "Walking / Standing", "color": "#a855f7"},
    {"key": "p", "id": "phone", "name": "Phone Use", "color": "#ef4444"},
    {"key": "c", "id": "call", "name": "Phone Call", "color": "#f97316"},
    {"key": "a", "id": "away", "name": "Away", "color": "#64748b"},
    {"key": "u", "id": "uncertain", "name": "Uncertain", "color": "#94a3b8"}
]

STUDENTS = ["Aagam", "Swayam", "Harshil", "Soham", "Unknown"]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ground-Truth Interval Labeling Tool</title>
<style>
  :root {
    --bg-main: #0f172a;
    --bg-card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-main);
    color: var(--text);
    padding: 16px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
  }
  header h1 { font-size: 1.25rem; font-weight: 600; color: var(--accent); }
  .header-controls { display: flex; gap: 12px; align-items: center; }
  .header-controls label { font-size: 0.85rem; color: var(--text-muted); }
  select, input {
    background: #0f172a;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
  }
  .main-layout {
    display: grid;
    grid-template-columns: 1fr 420px;
    gap: 16px;
    flex: 1;
    min-height: 0;
  }
  .player-section {
    display: flex;
    flex-direction: column;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    min-height: 0;
  }
  .video-container {
    flex: 1;
    background: #000;
    border-radius: 6px;
    overflow: hidden;
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  video { max-width: 100%; max-height: 100%; width: 100%; object-fit: contain; }
  .playback-bar {
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .timeline-container { display: flex; align-items: center; gap: 10px; }
  .timeline-slider { flex: 1; accent-color: var(--accent); cursor: pointer; }
  .time-badge { font-family: monospace; font-size: 0.95rem; font-weight: 600; color: #38bdf8; min-width: 160px; }
  .controls-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .btn-group { display: flex; gap: 6px; align-items: center; }
  button {
    background: #334155;
    border: 1px solid #475569;
    color: var(--text);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    cursor: pointer;
    font-weight: 500;
    transition: background 0.15s;
  }
  button:hover { background: #475569; }
  button.btn-primary { background: #0284c7; border-color: #0369a1; }
  button.btn-primary:hover { background: #0369a1; }
  button.btn-success { background: #16a34a; border-color: #15803d; }
  button.btn-success:hover { background: #15803d; }
  button.btn-danger { background: #dc2626; border-color: #b91c1c; }
  button.btn-danger:hover { background: #b91c1c; }
  
  .side-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
  }
  .card h2 { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 10px; }
  
  .class-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }
  .class-btn {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
    border: 1px solid transparent;
    cursor: pointer;
    background: #0f172a;
    color: var(--text);
    transition: all 0.15s;
  }
  .class-btn.active {
    border-color: #fff;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.2);
  }
  .class-btn .key {
    background: rgba(255,255,255,0.15);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: monospace;
    font-weight: bold;
  }
  
  .interval-creator {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .interval-times {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .time-input-box {
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
    text-align: center;
  }
  .time-input-box .label { font-size: 0.75rem; color: var(--text-muted); }
  .time-input-box .val { font-size: 1.1rem; font-family: monospace; font-weight: bold; color: var(--accent); margin: 4px 0; }
  
  .interval-list-container {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
  }
  .interval-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }
  .interval-table th {
    text-align: left;
    padding: 6px 8px;
    background: #0f172a;
    color: var(--text-muted);
    font-weight: 500;
    position: sticky;
    top: 0;
    border-bottom: 1px solid var(--border);
  }
  .interval-table td {
    padding: 6px 8px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .tag {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
  }
  .toast {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: #16a34a;
    color: #fff;
    padding: 10px 16px;
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
    font-size: 0.9rem;
    z-index: 100;
  }
  .toast.show { opacity: 1; }
</style>
</head>
<body>

<header>
  <h1>Computer Lab Ground-Truth Interval Labeling Tool</h1>
  <div class="header-controls">
    <label>Annotator*: <input type="text" id="annotatorInput" placeholder="Name required..." value="Reviewer" style="width: 130px;"></label>
    <label>Camera*: <input type="text" id="cameraInput" value="cam_0" style="width: 75px;"></label>
    <label>Video: <select id="videoSelect"></select></label>
    <button class="btn-success" onclick="exportCSV()">Export CSV</button>
  </div>
</header>

<div class="main-layout">
  <!-- Video Player Section -->
  <div class="player-section">
    <div class="video-container">
      <video id="videoPlayer" preload="auto"></video>
    </div>
    <div class="playback-bar">
      <div class="timeline-container">
        <span class="time-badge" id="timeDisplay">0.00s / 00:00.00</span>
        <input type="range" id="timelineSlider" class="timeline-slider" min="0" max="100" step="0.01" value="0">
        <span class="time-badge" id="durationDisplay" style="text-align: right;">0.00s</span>
      </div>
      <div class="controls-row">
        <div class="btn-group">
          <button onclick="stepFrame(-1)">&#9664; Prev ([,])</button>
          <button class="btn-primary" onclick="togglePlay()" id="playBtn">&#9658; Play (Space)</button>
          <button onclick="stepFrame(1)">Next ([.]) &#9654;</button>
        </div>
        <div class="btn-group">
          <label style="font-size: 0.8rem; color: var(--text-muted);">Speed:
            <select id="speedSelect" onchange="setSpeed(this.value)">
              <option value="0.25">0.25x</option>
              <option value="0.5">0.5x</option>
              <option value="1.0" selected>1.0x</option>
              <option value="2.0">2.0x</option>
            </select>
          </label>
        </div>
      </div>
    </div>
  </div>

  <!-- Side Control Panel -->
  <div class="side-panel">
    <!-- Student & Interval Creator -->
    <div class="card interval-creator">
      <h2>1. Target Person / Track & Boundaries</h2>
      <div style="display: flex; gap: 8px;">
        <select id="studentSelect" style="flex: 1;">
          <!-- Populated in JS -->
        </select>
        <input type="text" id="customPersonInput" placeholder="Or custom Track / Person ID..." style="flex: 1.2;">
      </div>
      <div class="interval-times">
        <div class="time-input-box">
          <div class="label">START TIME (Source Sec: [ or S)</div>
          <div class="val" id="startVal">--</div>
          <button onclick="setStart()">Set Current</button>
        </div>
        <div class="time-input-box">
          <div class="label">END TIME (Source Sec: ] or E)</div>
          <div class="val" id="endVal">--</div>
          <button onclick="setEnd()">Set Current</button>
        </div>
      </div>
      <input type="text" id="notesInput" placeholder="Notes (optional)...">
    </div>

    <!-- Activity Class Hotkeys -->
    <div class="card">
      <h2>2. Select Activity Class (Hotkey)</h2>
      <div class="class-grid" id="classGrid">
        <!-- Rendered in JS -->
      </div>
      <div style="margin-top: 10px;">
        <button class="btn-primary" style="width: 100%; padding: 10px;" onclick="saveInterval()">Save Interval (Enter)</button>
      </div>
    </div>

    <!-- Labeled Intervals Table -->
    <div class="card interval-list-container">
      <h2>Labeled Intervals (<span id="intervalCount">0</span>)</h2>
      <table class="interval-table">
        <thead>
          <tr>
            <th>Student</th>
            <th>Class</th>
            <th>Range</th>
            <th>Dur</th>
            <th></th>
          </tr>
        </thead>
        <tbody id="intervalTableBody">
        </tbody>
      </table>
    </div>
  </div>
</div>

<div class="toast" id="toast">Interval Saved!</div>

<script>
const CLASSES = """ + json.dumps(CLASSES) + """;
const STUDENTS = """ + json.dumps(STUDENTS) + """;

const video = document.getElementById("videoPlayer");
const timeDisplay = document.getElementById("timeDisplay");
const durationDisplay = document.getElementById("durationDisplay");
const slider = document.getElementById("timelineSlider");
const playBtn = document.getElementById("playBtn");
const videoSelect = document.getElementById("videoSelect");
const studentSelect = document.getElementById("studentSelect");
const classGrid = document.getElementById("classGrid");
const startVal = document.getElementById("startVal");
const endVal = document.getElementById("endVal");
const notesInput = document.getElementById("notesInput");
const annotatorInput = document.getElementById("annotatorInput");
const cameraInput = document.getElementById("cameraInput");
const customPersonInput = document.getElementById("customPersonInput");
const intervalTableBody = document.getElementById("intervalTableBody");
const intervalCount = document.getElementById("intervalCount");
const toast = document.getElementById("toast");

let selectedClass = "active_work";
let startTime = null;
let endTime = null;
let intervals = [];
let sourceFps = 30.0;

// Initialize Student Dropdown
STUDENTS.forEach(s => {
  const opt = document.createElement("option");
  opt.value = s;
  opt.textContent = s;
  studentSelect.appendChild(opt);
});

// Initialize Class Grid
CLASSES.forEach(c => {
  const btn = document.createElement("div");
  btn.className = `class-btn ${c.id === selectedClass ? 'active' : ''}`;
  btn.id = `class_btn_${c.id}`;
  btn.innerHTML = `<span style="color: ${c.color}; font-weight: 600;">${c.name}</span> <span class="key">${c.key.toUpperCase()}</span>`;
  btn.onclick = () => selectClass(c.id);
  classGrid.appendChild(btn);
});

function selectClass(cid) {
  selectedClass = cid;
  document.querySelectorAll(".class-btn").forEach(b => b.classList.remove("active"));
  const btn = document.getElementById(`class_btn_${cid}`);
  if (btn) btn.classList.add("active");
}

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2000);
}

// Load Videos
fetch("/api/videos")
  .then(r => r.json())
  .then(list => {
    list.forEach(v => {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      videoSelect.appendChild(opt);
    });
    if (list.length > 0) loadVideo(list[0]);
  });

videoSelect.onchange = (e) => loadVideo(e.target.value);

function loadVideo(vname) {
  video.src = `/video/${encodeURIComponent(vname)}`;
  loadIntervalsForVideo(vname);
}

video.onloadedmetadata = () => {
  slider.max = video.duration;
  durationDisplay.textContent = formatSec(video.duration);
};

video.ontimeupdate = () => {
  slider.value = video.currentTime;
  const t = video.currentTime;
  const f = Math.round(t * sourceFps);
  timeDisplay.textContent = `${t.toFixed(2)}s (f${f})`;
};

slider.oninput = (e) => {
  video.currentTime = parseFloat(e.target.value);
};

function formatSec(s) {
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(2);
  return `${m.toString().padStart(2, '0')}:${rem.padStart(5, '0')}`;
}

function togglePlay() {
  if (video.paused) {
    video.play();
    playBtn.innerHTML = "&#10074;&#10074; Pause (Space)";
  } else {
    video.pause();
    playBtn.innerHTML = "&#9658; Play (Space)";
  }
}

function stepFrame(delta) {
  video.pause();
  playBtn.innerHTML = "&#9658; Play (Space)";
  video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + delta * (1.0 / sourceFps)));
}

function setSpeed(v) {
  video.playbackRate = parseFloat(v);
}

function setStart() {
  startTime = video.currentTime;
  startVal.textContent = `${startTime.toFixed(2)}s`;
}

function setEnd() {
  endTime = video.currentTime;
  endVal.textContent = `${endTime.toFixed(2)}s`;
}

function saveInterval() {
  const annotator = (annotatorInput.value || "").trim();
  if (!annotator) {
    alert("Validation Error: 'Annotator' is a required field. Please enter your name.");
    annotatorInput.focus();
    return;
  }
  const person = (customPersonInput.value.trim() || studentSelect.value.trim());
  if (!person) {
    alert("Validation Error: 'Person (or track)' is a required field.");
    customPersonInput.focus();
    return;
  }
  const camera = (cameraInput.value || "cam_0").trim();
  if (!camera) {
    alert("Validation Error: 'Camera' is a required field.");
    cameraInput.focus();
    return;
  }
  if (startTime === null || endTime === null) {
    alert("Please set both Start Time ([) and End Time (]).");
    return;
  }
  const s = parseFloat(Math.min(startTime, endTime).toFixed(3));
  const e = parseFloat(Math.max(startTime, endTime).toFixed(3));
  if (e <= s) {
    alert("Validation Error: End time must be strictly greater than Start time.");
    return;
  }

  // Reject overlapping intervals for the same person on the same video
  const overlap = intervals.find(inv => {
    const pA = (inv.student || inv.person_id || "").toLowerCase();
    const pB = person.toLowerCase();
    if (pA !== pB) return false;
    const vA = inv.video_file || "";
    const vB = videoSelect.value || "";
    if (vA !== vB) return false;
    const inv_s = parseFloat(inv.source_video_start_s != null ? inv.source_video_start_s : inv.start_s);
    const inv_e = parseFloat(inv.source_video_end_s != null ? inv.source_video_end_s : inv.end_s);
    return Math.max(s, inv_s) < Math.min(e, inv_e);
  });

  if (overlap) {
    const o_s = parseFloat(overlap.source_video_start_s != null ? overlap.source_video_start_s : overlap.start_s).toFixed(2);
    const o_e = parseFloat(overlap.source_video_end_s != null ? overlap.source_video_end_s : overlap.end_s).toFixed(2);
    alert(`Validation Error: Overlapping interval rejected for person '${person}'!\n` +
          `New interval [${s.toFixed(2)}s - ${e.toFixed(2)}s] overlaps with existing interval [${o_s}s - ${o_e}s].\n` +
          `Intervals for the same person cannot overlap.`);
    return;
  }

  const item = {
    id: Date.now(),
    video_file: videoSelect.value,
    camera_id: camera,
    person_id: person,
    student: person,
    activity_label: selectedClass,
    source_video_start_s: s,
    source_video_end_s: e,
    start_s: s,
    end_s: e,
    start_frame: Math.round(s * sourceFps),
    end_frame: Math.round(e * sourceFps),
    duration_s: parseFloat((e - s).toFixed(3)),
    annotator: annotator,
    notes: notesInput.value || "",
    created_at: new Date().toISOString()
  };
  intervals.push(item);
  renderIntervals();
  saveIntervalsToServer();
  showToast("Interval Saved!");
  startTime = null;
  endTime = null;
  startVal.textContent = "--";
  endVal.textContent = "--";
  notesInput.value = "";
}

function deleteInterval(id) {
  intervals = intervals.filter(i => i.id !== id);
  renderIntervals();
  saveIntervalsToServer();
}

function renderIntervals() {
  intervalTableBody.innerHTML = "";
  intervalCount.textContent = intervals.length;
  intervals.forEach(inv => {
    const clsInfo = CLASSES.find(c => c.id === inv.activity_label) || { name: inv.activity_label, color: "#fff" };
    const pName = inv.student || inv.person_id || "Unknown";
    const cam = inv.camera_id || "cam_0";
    const s_sec = parseFloat(inv.source_video_start_s != null ? inv.source_video_start_s : inv.start_s).toFixed(1);
    const e_sec = parseFloat(inv.source_video_end_s != null ? inv.source_video_end_s : inv.end_s).toFixed(1);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${pName}</strong> <span style="font-size:0.75rem; color:#94a3b8;">(${cam})</span></td>
      <td><span class="tag" style="background: ${clsInfo.color}33; color: ${clsInfo.color}; border: 1px solid ${clsInfo.color}66;">${clsInfo.name}</span></td>
      <td>${s_sec}s - ${e_sec}s</td>
      <td>${parseFloat(inv.duration_s).toFixed(1)}s</td>
      <td><button class="btn-danger" style="padding: 2px 6px; font-size: 0.75rem;" onclick="deleteInterval(${inv.id})">&times;</button></td>
    `;
    intervalTableBody.appendChild(tr);
  });
}

function saveIntervalsToServer() {
  fetch("/api/save_intervals", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video: videoSelect.value, intervals: intervals })
  }).then(async r => {
    if (!r.ok) {
      const err = await r.json();
      alert("Server Validation Rejection: " + (err.message || "Failed to save intervals."));
    }
  });
}

function loadIntervalsForVideo(vname) {
  fetch(`/api/intervals?video=${encodeURIComponent(vname)}`)
    .then(r => r.json())
    .then(data => {
      intervals = data || [];
      renderIntervals();
    });
}

function exportCSV() {
  window.location.href = `/api/export_csv?video=${encodeURIComponent(videoSelect.value)}`;
}

// Global Keyboard Shortcuts
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  const key = e.key.toLowerCase();
  if (key === " ") {
    e.preventDefault();
    togglePlay();
  } else if (key === "," || key === "<") {
    e.preventDefault();
    stepFrame(-1);
  } else if (key === "." || key === ">") {
    e.preventDefault();
    stepFrame(1);
  } else if (key === "[" || key === "s") {
    e.preventDefault();
    setStart();
  } else if (key === "]" || key === "e") {
    e.preventDefault();
    setEnd();
  } else if (key === "enter") {
    e.preventDefault();
    saveInterval();
  } else {
    const matched = CLASSES.find(c => c.key === key);
    if (matched) {
      e.preventDefault();
      selectClass(matched.id);
    }
  }
});
</script>
</body>
</html>
"""

class IntervalLabelingServer(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path == "/api/videos":
            videos = sorted(glob.glob("*.mp4") + glob.glob("test_outputs/*.mp4"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps([os.path.basename(v) for v in videos]).encode("utf-8"))
            return

        elif path.startswith("/video/"):
            vname = urllib.parse.unquote(path[len("/video/"):])
            # Search candidate paths
            cand_paths = [vname, os.path.join("test_outputs", vname)]
            fpath = next((p for p in cand_paths if os.path.exists(p)), None)
            if fpath:
                self.serve_video_range(fpath)
            else:
                self.send_error(404, "Video not found")
            return

        elif path == "/api/intervals":
            qs = urllib.parse.parse_qs(parsed.query)
            vname = qs.get("video", [""])[0]
            csv_path = self.get_csv_path(vname)
            intervals = self.load_csv_intervals(csv_path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(intervals).encode("utf-8"))
            return

        elif path == "/api/export_csv":
            qs = urllib.parse.parse_qs(parsed.query)
            vname = qs.get("video", [""])[0]
            csv_path = self.get_csv_path(vname)
            if os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(csv_path)}"')
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404, "No intervals labeled for this video yet")
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save_intervals":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            vname = data.get("video", "video.mp4")
            intervals = data.get("intervals", [])
            
            # Validation step: reject overlapping intervals for same person/track and check required fields
            is_valid, err_msg = self.validate_intervals(intervals, vname)
            if not is_valid:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": err_msg}).encode("utf-8"))
                return

            csv_path = self.get_csv_path(vname)
            self.save_csv_intervals(csv_path, intervals)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            return
        self.send_error(404)

    def validate_intervals(self, intervals, vname):
        """
        Validates interval list:
        1. Required fields: person (or track), annotator, camera, source-video seconds.
        2. Rejects overlapping intervals for the same person on the same video.
        """
        from collections import defaultdict
        by_person = defaultdict(list)

        for idx, inv in enumerate(intervals):
            pid = (inv.get("person_id") or inv.get("student") or "").strip()
            if not pid:
                return False, f"Interval #{idx+1} is missing the required person (or track) field."
            ann = (inv.get("annotator") or "").strip()
            if not ann:
                return False, f"Interval #{idx+1} for '{pid}' is missing the required annotator field."
            cam = (inv.get("camera_id") or "").strip()
            if not cam:
                return False, f"Interval #{idx+1} for '{pid}' is missing the required camera field."

            try:
                s = float(inv.get("source_video_start_s") if inv.get("source_video_start_s") is not None else inv.get("start_s", 0.0))
                e = float(inv.get("source_video_end_s") if inv.get("source_video_end_s") is not None else inv.get("end_s", 0.0))
            except (ValueError, TypeError):
                return False, f"Interval #{idx+1} for '{pid}' has invalid numeric source-video seconds."

            if e <= s:
                return False, f"Interval #{idx+1} for '{pid}' has non-positive duration (end {e:.2f}s <= start {s:.2f}s)."

            norm_pid = pid.lower()
            for prev_s, prev_e, prev_id in by_person[norm_pid]:
                if max(s, prev_s) < min(e, prev_e):
                    return False, (
                        f"Overlapping intervals rejected for person '{pid}' on {vname}: "
                        f"Interval [{s:.2f}s - {e:.2f}s] overlaps with existing interval [{prev_s:.2f}s - {prev_e:.2f}s]."
                    )
            by_person[norm_pid].append((s, e, inv.get("id")))

        return True, "OK"

    def get_csv_path(self, vname: str) -> str:
        os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(vname))[0]
        return os.path.join(GROUND_TRUTH_DIR, f"{base}_intervals.csv")

    def load_csv_intervals(self, csv_path: str):
        if not os.path.exists(csv_path):
            return []
        items = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                s_sec = float(r.get("source_video_start_s") or r.get("start_s", 0.0))
                e_sec = float(r.get("source_video_end_s") or r.get("end_s", 0.0))
                p_name = r.get("person_id") or r.get("student", "")
                items.append({
                    "id": int(r.get("interval_id", 0)) or hash((p_name, s_sec)),
                    "video_file": r.get("video_file", ""),
                    "camera_id": r.get("camera_id", "cam_0"),
                    "person_id": p_name,
                    "student": p_name,
                    "activity_label": r.get("activity_label", ""),
                    "source_video_start_s": s_sec,
                    "source_video_end_s": e_sec,
                    "start_s": s_sec,
                    "end_s": e_sec,
                    "start_frame": int(r.get("start_frame", 0)),
                    "end_frame": int(r.get("end_frame", 0)),
                    "duration_s": float(r.get("duration_s", round(e_sec - s_sec, 3))),
                    "annotator": r.get("annotator", ""),
                    "notes": r.get("notes", ""),
                    "created_at": r.get("created_at", "")
                })
        return items

    def save_csv_intervals(self, csv_path: str, intervals):
        fieldnames = [
            "interval_id", "video_file", "camera_id", "person_id", "student", "activity_label",
            "source_video_start_s", "source_video_end_s", "start_s", "end_s", "start_frame", "end_frame",
            "duration_s", "annotator", "notes", "created_at"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for inv in intervals:
                s_sec = float(inv.get("source_video_start_s") if inv.get("source_video_start_s") is not None else inv.get("start_s", 0.0))
                e_sec = float(inv.get("source_video_end_s") if inv.get("source_video_end_s") is not None else inv.get("end_s", 0.0))
                p_name = inv.get("person_id") or inv.get("student", "")
                writer.writerow({
                    "interval_id": inv.get("id"),
                    "video_file": inv.get("video_file"),
                    "camera_id": inv.get("camera_id", "cam_0"),
                    "person_id": p_name,
                    "student": p_name,
                    "activity_label": inv.get("activity_label"),
                    "source_video_start_s": round(s_sec, 3),
                    "source_video_end_s": round(e_sec, 3),
                    "start_s": round(s_sec, 3),
                    "end_s": round(e_sec, 3),
                    "start_frame": inv.get("start_frame"),
                    "end_frame": inv.get("end_frame"),
                    "duration_s": round(inv.get("duration_s", e_sec - s_sec), 3),
                    "annotator": inv.get("annotator"),
                    "notes": inv.get("notes"),
                    "created_at": inv.get("created_at")
                })

    def serve_video_range(self, fpath: str):
        size = os.path.getsize(fpath)
        range_header = self.headers.get("Range")
        if range_header:
            bytes_unit, byte_range = range_header.split("=")
            start_str, end_str = byte_range.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else size - 1
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(fpath, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(fpath, "rb") as f:
                self.wfile.write(f.read())


def run_cli_labeler():
    print("=== Command-Line Ground-Truth Interval Labeler ===")
    vname = input("Enter video filename (default: new lab video.mp4): ").strip() or "new lab video.mp4"
    annotator = ""
    while not annotator:
        annotator = input("Enter your name (Annotator, REQUIRED): ").strip()
        if not annotator:
            print("Annotator name cannot be empty.")
    camera_id = input("Enter camera ID (default: cam_0): ").strip() or "cam_0"
    csv_path = os.path.join(GROUND_TRUTH_DIR, f"{os.path.splitext(vname)[0]}_intervals.csv")
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    print(f"Logging to: {csv_path}")

    # Load existing intervals to check for overlaps
    existing_intervals = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    existing_intervals.append({
                        "person_id": (r.get("person_id") or r.get("student", "")).strip().lower(),
                        "s": float(r.get("source_video_start_s") or r.get("start_s", 0.0)),
                        "e": float(r.get("source_video_end_s") or r.get("end_s", 0.0))
                    })
                except Exception:
                    pass

    while True:
        print("\nEnter interval details (or 'q' to quit):")
        person = input(f"Person / Track ID [{', '.join(STUDENTS)} or custom]: ").strip()
        if not person or person.lower() == 'q':
            break
        lbl = input("Activity Class: ").strip()
        if not lbl or lbl.lower() == 'q':
            break
        try:
            start_s = float(input("Start Time in source-video seconds: "))
            end_s = float(input("End Time in source-video seconds: "))
        except ValueError:
            print("[ERROR] Invalid numeric time.")
            continue
        if end_s <= start_s:
            print(f"[ERROR] End time ({end_s:.2f}s) must be strictly greater than start time ({start_s:.2f}s).")
            continue

        # Overlap check
        overlap = False
        for ex in existing_intervals:
            if ex["person_id"] == person.lower() and max(start_s, ex["s"]) < min(end_s, ex["e"]):
                print(f"[ERROR] Overlapping interval rejected for '{person}'! Overlaps with [{ex['s']:.2f}s - {ex['e']:.2f}s].")
                overlap = True
                break
        if overlap:
            continue

        notes = input("Notes: ").strip()
        item = {
            "interval_id": int(datetime.now().timestamp() * 1000),
            "video_file": vname,
            "camera_id": camera_id,
            "person_id": person,
            "student": person,
            "activity_label": lbl,
            "source_video_start_s": round(start_s, 3),
            "source_video_end_s": round(end_s, 3),
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "start_frame": int(start_s * 30.0),
            "end_frame": int(end_s * 30.0),
            "duration_s": round(end_s - start_s, 3),
            "annotator": annotator,
            "notes": notes,
            "created_at": datetime.now().isoformat()
        }
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(item.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(item)
        existing_intervals.append({"person_id": person.lower(), "s": start_s, "e": end_s})
        print(f"[SUCCESS] Logged interval [{start_s:.2f}s - {end_s:.2f}s] for '{person}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--cli", action="store_true")
    args = parser.parse_args()

    if args.cli:
        run_cli_labeler()
    else:
        server = HTTPServer(("0.0.0.0", args.port), IntervalLabelingServer)
        print(f"=======================================================")
        print(f"INTERVAL LABELING TOOL RUNNING ON http://127.0.0.1:{args.port}")
        print(f"Hotkeys: [W]ork [T]write [R]ead [D]iscuss [I]dle [K]walk [P]hone [C]all")
        print(f"Marks  : [ or S: Start | ] or E: End | Space: Play/Pause | Enter: Save")
        print(f"=======================================================")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down interval labeling server.")
