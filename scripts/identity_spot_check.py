"""
scripts/identity_spot_check.py
------------------------------
Human Review & Stratified Spot-Check Tool for Track/Identity Assignments across 100 frames.

Features:
- Stratified sampling across students (Aagam, Swayam, Harshil, Soham) and hidden-name frames
- Web-based review UI at http://127.0.0.1:5005 with hotkeys:
    [C] Mark Correct
    [W] Mark Wrong
    [U] Mark Unsure
    [<] / [A] Previous Frame
    [>] / [D] Next Frame
- CLI mode via --cli
- Real-time calculation and reporting of:
    - Reviewed count (n)
    - Correct, Wrong, Unsure counts
    - Error rate: p = Wrong / (Correct + Wrong)
    - 95% One-Sided Upper Confidence Bound (Wilson Score & Clopper-Pearson exact)
    - Per-Student Error Rates (Aagam, Swayam, Harshil, Soham, Hidden/Identifying)
    - Block Bootstrap 95% Confidence Interval (accounting for temporal autocorrelation)
    - Prominent Temporal Autocorrelation Caveat
- Saves review results to ground_truth/identity_spot_check_100.json
"""

import os
import sys
import json
import math
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
from collections import defaultdict
import numpy as np

SPOT_CHECK_DIR = "spot_check_frames"
ANNOTATIONS_FILE = os.path.join("ground_truth", "identity_spot_check_100.json")
PORT = 5005
KNOWN_STRATA = ["Aagam", "Swayam", "Harshil", "Soham", "Hidden/Identifying"]


def wilson_upper_bound(k: int, n: int, z: float = 1.6448536) -> float:
    """Computes one-sided 95% Wilson upper confidence bound."""
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))
    return round(min(1.0, max(0.0, center + spread)), 4)


def compute_block_bootstrap(reviewed_ann, block_size_s: float = 5.0, n_bootstraps: int = 1000):
    """
    Computes Block Bootstrap confidence intervals over temporal blocks
    to account for temporal autocorrelation across frames.
    """
    blocks = defaultdict(list)
    for a in reviewed_ann:
        lbl = a.get("label")
        if lbl in ("correct", "wrong"):
            t = float(a.get("timestamp_s", 0.0))
            b_id = int(t // block_size_s)
            val = 1 if lbl == "wrong" else 0
            blocks[b_id].append(val)

    block_ids = list(blocks.keys())
    m = len(block_ids)
    if m == 0:
        return {
            "block_size_s": block_size_s,
            "num_blocks": 0,
            "bootstrap_replicates": n_bootstraps,
            "mean": 0.0,
            "std_err": 0.0,
            "ci_95": [0.0, 0.0],
            "ci_95_pct": "0.00% - 0.00%",
            "upper_95": 0.0,
            "upper_95_pct": "0.00%"
        }

    total_errors = sum(sum(v) for v in blocks.values())
    if total_errors == 0:
        # Zero errors found: avoid degenerate [0.0, 0.0] bootstrap interval.
        # Use exact 1-sided 95% upper bound treating m independent 5s blocks as effective sample size:
        # p_upper = 1 - (alpha)^(1 / m) where alpha = 0.05
        exact_block_upper = 1.0 - (0.05 ** (1.0 / max(1, m)))
        rule3_upper = min(1.0, 3.0 / max(1, m))
        return {
            "block_size_s": block_size_s,
            "num_blocks": m,
            "effective_n_blocks": m,
            "bootstrap_replicates": n_bootstraps,
            "mean": 0.0,
            "mean_pct": "0.00%",
            "std_err": 0.0,
            "ci_95": [0.0, round(float(exact_block_upper), 4)],
            "ci_95_pct": f"0.00% - {exact_block_upper * 100:.2f}% (Exact Block Upper)",
            "upper_95": round(float(exact_block_upper), 4),
            "upper_95_pct": f"{exact_block_upper * 100:.2f}%",
            "rule_of_three_upper_pct": f"{rule3_upper * 100:.2f}%",
            "is_zero_error": True,
            "degenerate_bootstrap_prevented": True,
            "method": "Exact Binomial Upper Bound on M Independent 5s Blocks (Non-Degenerate)"
        }

    rng = np.random.RandomState(42)
    boot_rates = []
    for _ in range(n_bootstraps):
        sampled_b = rng.choice(block_ids, size=m, replace=True)
        decisions = []
        for b in sampled_b:
            decisions.extend(blocks[b])
        if decisions:
            boot_rates.append(float(np.mean(decisions)))
        else:
            boot_rates.append(0.0)

    b_arr = np.array(boot_rates)
    b_mean = float(np.mean(b_arr))
    b_std = float(np.std(b_arr))
    ci_low, ci_high = np.percentile(b_arr, [2.5, 97.5])
    upper_95 = float(np.percentile(b_arr, 95.0))

    return {
        "block_size_s": block_size_s,
        "num_blocks": m,
        "bootstrap_replicates": n_bootstraps,
        "mean": round(b_mean, 4),
        "mean_pct": f"{b_mean * 100:.2f}%",
        "std_err": round(b_std, 4),
        "ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "ci_95_pct": f"{ci_low * 100:.2f}% - {ci_high * 100:.2f}%",
        "upper_95": round(upper_95, 4),
        "upper_95_pct": f"{upper_95 * 100:.2f}%"
    }


def compute_error_metrics(annotations, total_frames=100):
    """
    Computes overall error metrics, per-student error rates, and block bootstrap statistics.
    """
    meta = load_metadata()
    strata_counts = defaultdict(int)
    meta_by_file = {}
    for m in meta:
        fn = m.get("image_file")
        strat = m.get("stratum", "Unassigned")
        meta_by_file[fn] = m
        strata_counts[strat] += 1

    reviewed = [a for a in annotations.values() if a.get("label") in ("correct", "wrong", "unsure")]
    n_total = len(reviewed)
    n_correct = sum(1 for a in reviewed if a.get("label") == "correct")
    n_wrong = sum(1 for a in reviewed if a.get("label") == "wrong")
    n_unsure = sum(1 for a in reviewed if a.get("label") == "unsure")

    n_eval = n_correct + n_wrong
    if n_eval == 0:
        p_hat = 0.0
        wilson_upper = 0.0
        exact_upper = 0.0
    else:
        p_hat = n_wrong / n_eval
        wilson_upper = wilson_upper_bound(n_wrong, n_eval)
        try:
            from scipy.stats import beta
            exact_upper = 1.0 if n_wrong == n_eval else float(beta.ppf(0.95, n_wrong + 1, n_eval - n_wrong))
        except Exception:
            exact_upper = min(1.0, 3.0 / n_eval) if n_wrong == 0 else wilson_upper

    # Per-Student / Per-Stratum Error Breakdown
    per_student = {}
    for strat in KNOWN_STRATA:
        # Match either by sample stratum tag or by active names / tracks
        strat_anns = []
        for a in reviewed:
            fn = a.get("image_file")
            sample_strat = meta_by_file.get(fn, {}).get("stratum")
            # If explicit stratum matches OR person is in active_names
            act_names = meta_by_file.get(fn, {}).get("active_names", [])
            in_stratum = (sample_strat == strat) or (strat in act_names)
            if strat == "Hidden/Identifying":
                in_stratum = (sample_strat == strat) or any("Identifying" in n or "Unknown" in n for n in act_names)

            if in_stratum:
                strat_anns.append(a)

        s_total = len(strat_anns)
        s_corr = sum(1 for a in strat_anns if a.get("label") == "correct")
        s_wrng = sum(1 for a in strat_anns if a.get("label") == "wrong")
        s_uns = sum(1 for a in strat_anns if a.get("label") == "unsure")
        s_eval = s_corr + s_wrng
        s_rate = (s_wrng / s_eval) if s_eval > 0 else 0.0
        s_upper = wilson_upper_bound(s_wrng, s_eval)

        per_student[strat] = {
            "stratum_name": strat,
            "stratum_target_frames": strata_counts.get(strat, 20),
            "reviewed": s_total,
            "evaluated": s_eval,
            "correct": s_corr,
            "wrong": s_wrng,
            "unsure": s_uns,
            "error_rate": round(s_rate, 4),
            "error_rate_pct": f"{s_rate * 100:.2f}%",
            "wilson_upper_95": s_upper,
            "wilson_upper_95_pct": f"{s_upper * 100:.2f}%"
        }

    # Block Bootstrap calculation
    bootstrap_stats = compute_block_bootstrap(reviewed, block_size_s=5.0, n_bootstraps=1000)

    caveat_text = (
        "Temporal Autocorrelation Caveat: Video frames sampled from continuous tracking sequences "
        "exhibit serial correlation. Standard i.i.d. intervals (Wilson, Clopper-Pearson) assume independent "
        "Bernoulli trials and underestimate true variance when errors persist across consecutive frames. "
        "The Block Bootstrap accounts for serial dependency by resampling contiguous 5.0-second time blocks."
    )

    return {
        "reviewed": n_total,
        "total_samples": total_frames,
        "evaluated": n_eval,
        "correct": n_correct,
        "wrong": n_wrong,
        "unsure": n_unsure,
        "error_rate": round(p_hat, 4),
        "error_rate_pct": f"{p_hat * 100:.2f}%",
        "wilson_upper_95": round(wilson_upper, 4),
        "wilson_upper_95_pct": f"{wilson_upper * 100:.2f}%",
        "exact_upper_95": round(exact_upper, 4),
        "exact_upper_95_pct": f"{exact_upper * 100:.2f}%",
        "per_student": per_student,
        "block_bootstrap": bootstrap_stats,
        "temporal_autocorrelation_caveat": caveat_text
    }


def load_metadata():
    meta_path = os.path.join(SPOT_CHECK_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_annotations():
    if os.path.exists(ANNOTATIONS_FILE):
        try:
            with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_annotations(data):
    os.makedirs(os.path.dirname(ANNOTATIONS_FILE), exist_ok=True)
    with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Identity Stratified Spot-Check Tool (100 Frames)</title>
<style>
  :root {
    --bg-main: #0b0f19;
    --bg-card: #151d2f;
    --border: #24324d;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --correct: #10b981;
    --wrong: #ef4444;
    --unsure: #f59e0b;
    --stratum-badge: #818cf8;
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
  .layout {
    display: grid;
    grid-template-columns: 1fr 420px;
    gap: 16px;
    flex: 1;
    min-height: 0;
  }
  .frame-view {
    display: flex;
    flex-direction: column;
    background: #000;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
    position: relative;
    align-items: center;
    justify-content: center;
    min-height: 0;
  }
  .frame-view img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 4px;
  }
  .side-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
    overflow-y: auto;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }
  .card h2 { font-size: 0.95rem; font-weight: 600; margin-bottom: 10px; color: var(--accent); }
  .btn-group { display: flex; gap: 8px; margin-top: 10px; }
  button {
    flex: 1;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid transparent;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    transition: all 0.15s;
    color: #fff;
  }
  button.correct { background: #065f46; border-color: var(--correct); }
  button.correct:hover, button.correct.active { background: var(--correct); color: #000; }
  button.wrong { background: #7f1d1d; border-color: var(--wrong); }
  button.wrong:hover, button.wrong.active { background: var(--wrong); color: #fff; }
  button.unsure { background: #78350f; border-color: var(--unsure); }
  button.unsure:hover, button.unsure.active { background: var(--unsure); color: #000; }
  button.nav-btn { background: #1e293b; border-color: #334155; }
  button.nav-btn:hover { background: #334155; }

  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 8px;
  }
  .stat-box {
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    text-align: center;
  }
  .stat-box .num { font-size: 1.25rem; font-weight: bold; font-family: monospace; }
  .stat-box .lbl { font-size: 0.72rem; color: var(--text-muted); margin-top: 2px; }

  .highlight-card {
    border: 1px solid var(--accent);
    background: rgba(56, 189, 248, 0.04);
  }
  .caveat-box {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid #f59e0b44;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.72rem;
    color: #fbbf24;
    line-height: 1.35;
    margin-top: 8px;
  }
  .key-hint {
    background: rgba(255,255,255,0.15);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: monospace;
  }
  .stratum-pill {
    background: rgba(129, 140, 248, 0.2);
    color: var(--stratum-badge);
    border: 1px solid var(--stratum-badge);
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .track-list {
    margin-top: 8px;
    font-size: 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .track-row {
    background: #0f172a;
    padding: 6px 10px;
    border-radius: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .badge {
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: bold;
  }
  table.per-student-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.75rem;
    margin-top: 8px;
  }
  table.per-student-tbl th, table.per-student-tbl td {
    padding: 5px 6px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  table.per-student-tbl th {
    color: var(--text-muted);
    font-weight: 600;
  }
</style>
</head>
<body>
<header>
  <h1>Identity Stratified Spot-Check (100 Frames)</h1>
  <div style="font-size: 0.85rem; color: var(--text-muted);">
    Hotkeys: <span class="key-hint">C</span> Correct &nbsp;|&nbsp; <span class="key-hint">W</span> Wrong &nbsp;|&nbsp; <span class="key-hint">U</span> Unsure &nbsp;|&nbsp; <span class="key-hint">&larr;</span> Prev &nbsp;|&nbsp; <span class="key-hint">&rarr;</span> Next
  </div>
</header>

<div class="layout">
  <div class="frame-view">
    <img id="frameImg" src="" alt="Spot Check Frame">
  </div>

  <div class="side-panel">
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <h2>Sample <span id="currentIdx">1</span> / <span id="totalIdx">100</span></h2>
        <span class="stratum-pill" id="stratumBadge">Stratum</span>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 8px;">
        <span id="frameFilename">frame_00000.jpg</span>
        <span id="timestampBadge" style="font-family: monospace; color: var(--accent);">0.00s</span>
      </div>

      <div class="btn-group">
        <button class="correct" id="btnCorrect" onclick="setDecision('correct')"><span class="key-hint">C</span> Correct</button>
        <button class="wrong" id="btnWrong" onclick="setDecision('wrong')"><span class="key-hint">W</span> Wrong</button>
        <button class="unsure" id="btnUnsure" onclick="setDecision('unsure')"><span class="key-hint">U</span> Unsure</button>
      </div>

      <div class="btn-group" style="margin-top: 8px;">
        <button class="nav-btn" onclick="prevFrame()">&larr; Prev (<span class="key-hint">&larr;</span>)</button>
        <button class="nav-btn" onclick="nextFrame()">Next (<span class="key-hint">&rarr;</span>) &rarr;</button>
      </div>
    </div>

    <div class="card">
      <h2>Active Tracks in Frame</h2>
      <div class="track-list" id="trackList">
        <div style="color: var(--text-muted); font-size: 0.8rem;">Loading tracks...</div>
      </div>
    </div>

    <div class="card highlight-card">
      <h2>Statistical Summary (Wilson & Block Bootstrap)</h2>
      <div class="stat-grid">
        <div class="stat-box">
          <div class="num" id="statReviewed">0 / 100</div>
          <div class="lbl">Reviewed</div>
        </div>
        <div class="stat-box">
          <div class="num" id="statErrorRate" style="color: var(--wrong);">0.0%</div>
          <div class="lbl">Sample Error Rate</div>
        </div>
        <div class="stat-box">
          <div class="num" id="statUpper95" style="color: var(--accent);">0.0%</div>
          <div class="lbl">95% Upper Bound (Wilson)</div>
        </div>
        <div class="stat-box">
          <div class="num" id="statBootstrapCI" style="color: #a78bfa; font-size: 1.05rem;">0.00% - 0.00%</div>
          <div class="lbl">95% Block Bootstrap CI</div>
        </div>
      </div>

      <div class="caveat-box">
        <strong>Temporal Autocorrelation Caveat:</strong>
        Video frames sampled from continuous tracks exhibit serial correlation. Standard Wilson / Exact intervals assume independent Bernoulli trials and underestimate true variance. Block Bootstrap accounts for serial dependency across contiguous 5s blocks.
      </div>

      <div style="display: flex; justify-content: space-around; font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">
        <span>Correct: <strong id="statCorrect" style="color: var(--correct);">0</strong></span>
        <span>Wrong: <strong id="statWrong" style="color: var(--wrong);">0</strong></span>
        <span>Unsure: <strong id="statUnsure" style="color: var(--unsure);">0</strong></span>
      </div>
    </div>

    <div class="card">
      <h2>Per-Student Error Rate Breakdown</h2>
      <table class="per-student-tbl">
        <thead>
          <tr>
            <th>Stratum / Student</th>
            <th>Eval</th>
            <th>Wrong</th>
            <th>Error %</th>
            <th>95% Upper</th>
          </tr>
        </thead>
        <tbody id="studentBreakdownBody">
          <tr><td colspan="5" style="color: var(--text-muted);">No annotations yet.</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
let samples = [];
let annotations = {};
let currentIndex = 0;

async function init() {
  const metaRes = await fetch("/api/metadata");
  samples = await metaRes.json();
  const annRes = await fetch("/api/annotations");
  annotations = await annRes.json();

  document.getElementById("totalIdx").textContent = samples.length || 100;
  loadCurrentSample();
  updateStats();
}

function loadCurrentSample() {
  if (!samples || samples.length === 0) return;
  const s = samples[currentIndex];
  document.getElementById("currentIdx").textContent = currentIndex + 1;
  document.getElementById("timestampBadge").textContent = `${s.timestamp_s.toFixed(2)}s (f${s.frame_idx})`;
  document.getElementById("frameFilename").textContent = s.image_file;
  document.getElementById("stratumBadge").textContent = s.stratum || "Stratum";
  document.getElementById("frameImg").src = `/frame_image/${s.image_file}`;

  // Update decision buttons
  const ann = annotations[s.image_file];
  document.getElementById("btnCorrect").classList.toggle("active", ann && ann.label === "correct");
  document.getElementById("btnWrong").classList.toggle("active", ann && ann.label === "wrong");
  document.getElementById("btnUnsure").classList.toggle("active", ann && ann.label === "unsure");

  // Render tracks
  const trackList = document.getElementById("trackList");
  trackList.innerHTML = "";
  if (s.tracks && s.tracks.length > 0) {
    s.tracks.forEach(t => {
      const row = document.createElement("div");
      row.className = "track-row";
      let col = "#94a3b8";
      if (t.status === "LOCKED") col = "#10b981";
      else if (t.status === "ASSIGNED") col = "#38bdf8";
      else if (t.status === "IDENTIFYING") col = "#eab308";

      row.innerHTML = `
        <div>
          <strong style="color: ${col};">${t.display_name}</strong>
          <span style="font-size: 0.75rem; color: #94a3b8;"> (Track ${t.track_id})</span>
        </div>
        <div style="font-family: monospace; font-size: 0.75rem;">
          <span class="badge" style="background: ${col}22; color: ${col}; border: 1px solid ${col}66;">${t.status}</span>
          <span>aff:${(t.affinity || 0.0).toFixed(2)} mar:${(t.margin || 0.0).toFixed(2)}</span>
        </div>
      `;
      trackList.appendChild(row);
    });
  } else {
    trackList.innerHTML = "<div style='color: #94a3b8;'>No active tracks in this frame.</div>";
  }
}

async function setDecision(label) {
  if (!samples || samples.length === 0) return;
  const s = samples[currentIndex];
  annotations[s.image_file] = {
    image_file: s.image_file,
    frame_idx: s.frame_idx,
    timestamp_s: s.timestamp_s,
    stratum: s.stratum || "Unassigned",
    label: label,
    tracks: s.tracks,
    reviewed_at: new Date().toISOString()
  };

  await fetch("/api/save_annotation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_file: s.image_file, record: annotations[s.image_file] })
  });

  loadCurrentSample();
  updateStats();
  if (currentIndex < samples.length - 1) {
    setTimeout(nextFrame, 150);
  }
}

function prevFrame() {
  if (currentIndex > 0) {
    currentIndex--;
    loadCurrentSample();
  }
}

function nextFrame() {
  if (currentIndex < samples.length - 1) {
    currentIndex++;
    loadCurrentSample();
  }
}

async function updateStats() {
  const res = await fetch("/api/stats");
  const stats = await res.json();
  document.getElementById("statReviewed").textContent = `${stats.reviewed} / ${stats.total_samples}`;
  document.getElementById("statErrorRate").textContent = stats.error_rate_pct;
  document.getElementById("statUpper95").textContent = `${stats.wilson_upper_95_pct}`;
  if (stats.block_bootstrap) {
    document.getElementById("statBootstrapCI").textContent = stats.block_bootstrap.ci_95_pct;
  }
  document.getElementById("statCorrect").textContent = stats.correct;
  document.getElementById("statWrong").textContent = stats.wrong;
  document.getElementById("statUnsure").textContent = stats.unsure;

  // Render per-student breakdown table
  const tbody = document.getElementById("studentBreakdownBody");
  if (stats.per_student && Object.keys(stats.per_student).length > 0) {
    tbody.innerHTML = "";
    for (const [sname, sinfo] of Object.entries(stats.per_student)) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${sname}</strong></td>
        <td>${sinfo.evaluated}</td>
        <td style="color: var(--wrong);">${sinfo.wrong}</td>
        <td>${sinfo.error_rate_pct}</td>
        <td style="color: var(--accent);">${sinfo.wilson_upper_95_pct}</td>
      `;
      tbody.appendChild(tr);
    }
  }
}

window.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === "c") {
    e.preventDefault();
    setDecision("correct");
  } else if (k === "w") {
    e.preventDefault();
    setDecision("wrong");
  } else if (k === "u") {
    e.preventDefault();
    setDecision("unsure");
  } else if (k === "arrowleft" || k === "a") {
    e.preventDefault();
    prevFrame();
  } else if (k === "arrowright" || k === "d") {
    e.preventDefault();
    nextFrame();
  }
});

init();
</script>
</body>
</html>
"""


class SpotCheckServer(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        elif path == "/api/metadata":
            meta = load_metadata()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(meta).encode("utf-8"))
            return

        elif path == "/api/annotations":
            ann = load_annotations()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(ann).encode("utf-8"))
            return

        elif path == "/api/stats":
            ann = load_annotations()
            meta = load_metadata()
            stats = compute_error_metrics(ann, total_frames=len(meta) or 100)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode("utf-8"))
            return

        elif path.startswith("/frame_image/"):
            fname = urllib.parse.unquote(path[len("/frame_image/"):])
            fpath = os.path.join(SPOT_CHECK_DIR, fname)
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Frame image not found")
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save_annotation":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            img_file = payload.get("image_file")
            rec = payload.get("record")
            ann = load_annotations()
            ann[img_file] = rec
            save_annotations(ann)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            return
        self.send_error(404)


def run_cli_spot_check():
    meta = load_metadata()
    if not meta:
        print("[ERROR] No metadata found in spot_check_frames/metadata.json. Run measurement first.")
        return
    ann = load_annotations()
    total = len(meta)

    print("=" * 80)
    print(f"IDENTITY 100-FRAME STRATIFIED SPOT-CHECK CLI (Total Frames: {total})")
    print("Commands: [c]orrect, [w]rong, [u]nsure, [p]revious, [s]tats, [q]uit")
    print("=" * 80)

    idx = 0
    while idx < total:
        s = meta[idx]
        fname = s["image_file"]
        cur_ann = ann.get(fname, {}).get("label", "unreviewed")
        stratum = s.get("stratum", "Unassigned")
        print(f"\n[{idx+1}/{total}] Frame {s['frame_idx']} ({s['timestamp_s']:.2f}s) | Stratum: {stratum} | Image: {fname} | Status: {cur_ann.upper()}")
        print("Tracks:")
        for t in s.get("tracks", []):
            print(f"  - {t['display_name']:<16} (Track {t['track_id']:<2}, {t['status']:<11}, aff={t.get('affinity', 0.0):.2f}, mar={t.get('margin', 0.0):.2f})")

        cmd = input("Decision [c/w/u/p/s/q]: ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "p":
            idx = max(0, idx - 1)
            continue
        elif cmd == "s":
            stats = compute_error_metrics(ann, total_frames=total)
            print("\nCurrent Metrics:")
            print(f"  Reviewed: {stats['reviewed']}/{stats['total_samples']}")
            print(f"  Correct: {stats['correct']}, Wrong: {stats['wrong']}, Unsure: {stats['unsure']}")
            print(f"  Sample Error Rate: {stats['error_rate_pct']}")
            print(f"  95% Upper Bound (Wilson): {stats['wilson_upper_95_pct']}")
            if stats.get("block_bootstrap"):
                print(f"  95% Block Bootstrap CI: {stats['block_bootstrap']['ci_95_pct']} (Mean: {stats['block_bootstrap']['mean_pct']})")
            print("\n  Per-Student Breakdown:")
            for sname, sinfo in stats.get("per_student", {}).items():
                print(f"    - {sname:<20}: Eval {sinfo['evaluated']}, Wrong {sinfo['wrong']}, Error {sinfo['error_rate_pct']}, 95% Upper {sinfo['wilson_upper_95_pct']}")
            print(f"\n  NOTE: {stats.get('temporal_autocorrelation_caveat')}\n")
            continue
        elif cmd in ("c", "w", "u"):
            lbl = "correct" if cmd == "c" else "wrong" if cmd == "w" else "unsure"
            ann[fname] = {
                "image_file": fname,
                "frame_idx": s["frame_idx"],
                "timestamp_s": s["timestamp_s"],
                "stratum": stratum,
                "label": lbl,
                "tracks": s.get("tracks", [])
            }
            save_annotations(ann)
            idx += 1
        else:
            print("Invalid input.")

    stats = compute_error_metrics(ann, total_frames=total)
    print("\n" + "=" * 80)
    print("FINAL SPOT-CHECK SUMMARY:")
    print(f"  Reviewed: {stats['reviewed']} / {stats['total_samples']}")
    print(f"  Correct: {stats['correct']}, Wrong: {stats['wrong']}, Unsure: {stats['unsure']}")
    print(f"  Sample Error Rate: {stats['error_rate_pct']}")
    print(f"  95% Upper Bound (Wilson): {stats['wilson_upper_95_pct']}")
    if stats.get("block_bootstrap"):
        print(f"  95% Block Bootstrap CI: {stats['block_bootstrap']['ci_95_pct']}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        ann = load_annotations()
        meta = load_metadata()
        stats = compute_error_metrics(ann, total_frames=len(meta) or 100)
        print(json.dumps(stats, indent=2))
    elif args.cli:
        run_cli_spot_check()
    else:
        server = HTTPServer(("0.0.0.0", args.port), SpotCheckServer)
        print(f"=======================================================")
        print(f"IDENTITY SPOT-CHECK TOOL RUNNING ON http://127.0.0.1:{args.port}")
        print(f"Hotkeys: [C] Correct, [W] Wrong, [U] Unsure, [<-] Prev, [->] Next")
        print(f"Stratified frames: 100 frames across 5 strata (Aagam, Swayam, Harshil, Soham, Hidden)")
        print(f"Block Bootstrap: 5.0s blocks to handle temporal autocorrelation")
        print(f"=======================================================")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down spot-check tool.")
