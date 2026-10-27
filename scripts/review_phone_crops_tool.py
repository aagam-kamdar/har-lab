"""
review_phone_crops_tool.py
Human review and labeling tool for the 640 hand-crop phone candidates:
- Interactive Web UI: Displays crop image, metadata, and 1-click / keyboard shortcut labeling:
    [1] or 'P' -> Phone
    [2] or 'N' -> No-Phone
    [3] or 'U' -> Unsure
- Automatically persists annotations directly to test_outputs/phone_dataset/annotations_template.csv.
- CLI fallback mode available via --cli.
MANDATORY CONSTRAINT: Pure labeling tool. Zero training is executed.
"""

import os
import sys
import csv
import argparse
from typing import Dict, List
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CSV_PATH = "test_outputs/phone_dataset/annotations_template.csv"
IMG_DIR = "test_outputs/phone_dataset/images"

app = FastAPI(title="Phone Crop Human Review Tool")

def load_records() -> List[Dict]:
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_records(records: List[Dict]):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

class LabelUpdate(BaseModel):
    crop_id: str
    label: str # 'phone', 'no_phone', 'unsure'
    notes: str = ""

@app.get("/api/crops")
def get_crops():
    records = load_records()
    labeled = sum(1 for r in records if r.get("human_label"))
    return {
        "total": len(records),
        "labeled": labeled,
        "unlabeled": len(records) - labeled,
        "records": records
    }

@app.post("/api/label")
def post_label(update: LabelUpdate):
    records = load_records()
    found = False
    for r in records:
        if r["crop_id"] == update.crop_id:
            r["human_label"] = update.label
            if update.notes:
                r["notes"] = update.notes
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Crop ID not found")
    save_records(records)
    return {"status": "ok", "crop_id": update.crop_id, "label": update.label}

@app.get("/images/{filename}")
def get_crop_image(filename: str):
    path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Phone Crop Human Labeling Tool</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; display: flex; flex-direction: column; align-items: center; }
    .card { background: #1e293b; border-radius: 12px; padding: 24px; max-width: 680px; width: 100%; box-shadow: 0 10px 25px rgba(0,0,0,0.5); text-align: center; border: 1px solid #334155; }
    h1 { margin-top: 0; font-size: 22px; color: #38bdf8; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; margin-bottom: 16px; background: #3b82f6; color: white; }
    .img-box { margin: 16px auto; width: 280px; height: 280px; border: 2px dashed #475569; border-radius: 8px; display: flex; align-items: center; justify-content: center; background: #000; overflow: hidden; }
    .img-box img { width: 100%; height: 100%; object-fit: contain; }
    .meta { font-size: 14px; color: #94a3b8; margin-bottom: 20px; line-height: 1.6; }
    .btn-group { display: flex; gap: 12px; justify-content: center; margin-bottom: 20px; }
    button { padding: 12px 24px; border-radius: 8px; border: none; font-size: 15px; font-weight: bold; cursor: pointer; transition: all 0.15s ease; }
    button:hover { transform: translateY(-2px); }
    .btn-phone { background: #10b981; color: white; }
    .btn-nophone { background: #ef4444; color: white; }
    .btn-unsure { background: #f59e0b; color: white; }
    .btn-nav { background: #475569; color: white; padding: 8px 16px; }
    .progress-bar-wrap { width: 100%; background: #334155; border-radius: 6px; height: 10px; margin: 20px 0; overflow: hidden; }
    .progress-bar-fill { height: 100%; background: #38bdf8; width: 0%; transition: width 0.3s ease; }
    .kbd { background: #334155; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 12px; }
    .status-text { font-size: 14px; font-weight: 600; color: #38bdf8; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Phone Hand-Crop Human Annotation Tool</h1>
    <div class="badge" id="lblBadge">Candidate Proposal</div>
    
    <div class="img-box">
      <img id="cropImg" src="" alt="Hand Crop">
    </div>

    <div class="meta" id="metaText">Loading crops...</div>

    <div class="btn-group">
      <button class="btn-phone" onclick="submitLabel('phone')">[1] Phone</button>
      <button class="btn-nophone" onclick="submitLabel('no_phone')">[2] No-Phone</button>
      <button class="btn-unsure" onclick="submitLabel('unsure')">[3] Unsure</button>
    </div>

    <div class="btn-group">
      <button class="btn-nav" onclick="prevCrop()">&larr; Previous</button>
      <button class="btn-nav" onclick="nextCrop()">Next &rarr;</button>
    </div>

    <div class="progress-bar-wrap">
      <div class="progress-bar-fill" id="progFill"></div>
    </div>
    <div class="status-text" id="statusCount">0 / 640 Reviewed (0%)</div>

    <p style="font-size: 12px; color: #64748b; margin-top: 16px;">
      Hotkeys: <span class="kbd">1</span> or <span class="kbd">P</span> = Phone, 
      <span class="kbd">2</span> or <span class="kbd">N</span> = No-Phone, 
      <span class="kbd">3</span> or <span class="kbd">U</span> = Unsure, 
      <span class="kbd">&larr;</span>/<span class="kbd">&rarr;</span> = Navigate
    </p>
  </div>

  <script>
    let crops = [];
    let curIdx = 0;

    async function loadData() {
      const res = await fetch('/api/crops');
      const data = await res.json();
      crops = data.records;
      // Find first unlabeled crop
      const firstUn = crops.findIndex(c => !c.human_label);
      curIdx = firstUn !== -1 ? firstUn : 0;
      render();
    }

    function render() {
      if (!crops.length) return;
      const c = crops[curIdx];
      document.getElementById('cropImg').src = '/images/' + c.image_file;
      document.getElementById('lblBadge').innerText = c.candidate_class;
      document.getElementById('lblBadge').style.background = c.candidate_class.includes('POS') ? '#10b981' : '#64748b';
      
      const curLbl = c.human_label ? `<strong>Current: ${c.human_label.toUpperCase()}</strong>` : '<em>Unlabeled</em>';
      document.getElementById('metaText').innerHTML = `
        <strong>ID:</strong> ${c.crop_id} | <strong>File:</strong> ${c.image_file}<br>
        <strong>Category:</strong> ${c.heuristic_category} | <strong>Time:</strong> ${c.time_s}s (f${c.frame})<br>
        <strong>Status:</strong> ${curLbl}
      `;

      const labeledCount = crops.filter(x => x.human_label).length;
      const pct = Math.round(labeledCount / crops.length * 100);
      document.getElementById('progFill').style.width = pct + '%';
      document.getElementById('statusCount').innerText = `${curIdx + 1} of ${crops.length} | Total Reviewed: ${labeledCount}/${crops.length} (${pct}%)`;
    }

    async function submitLabel(val) {
      if (!crops.length) return;
      const c = crops[curIdx];
      c.human_label = val;
      await fetch('/api/label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({crop_id: c.crop_id, label: val})
      });
      nextCrop();
    }

    function nextCrop() {
      if (curIdx < crops.length - 1) {
        curIdx++;
        render();
      }
    }

    function prevCrop() {
      if (curIdx > 0) {
        curIdx--;
        render();
      }
    }

    window.addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      if (k === '1' || k === 'p') submitLabel('phone');
      else if (k === '2' || k === 'n') submitLabel('no_phone');
      else if (k === '3' || k === 'u') submitLabel('unsure');
      else if (e.key === 'ArrowRight') nextCrop();
      else if (e.key === 'ArrowLeft') prevCrop();
    });

    loadData();
  </script>
</body>
</html>
    """

def run_cli_mode():
    records = load_records()
    if not records:
        print("[ERROR] No crops found.")
        return
    print("=" * 65)
    print("CLI PHONE CROP REVIEW MODE (Type 1=phone, 2=no_phone, 3=unsure, q=quit)")
    print("=" * 65)
    for idx, r in enumerate(records):
        if r.get("human_label"):
            continue
        print(f"\n[{idx+1}/{len(records)}] Crop ID: {r['crop_id']} | Image: {r['image_file']} | Category: {r['heuristic_category']}")
        ans = input("Label ([1] Phone, [2] No-Phone, [3] Unsure, [q] Quit): ").strip().lower()
        if ans == 'q':
            break
        elif ans in ('1', 'phone', 'p'):
            r["human_label"] = "phone"
        elif ans in ('2', 'no_phone', 'n', '0'):
            r["human_label"] = "no_phone"
        elif ans in ('3', 'unsure', 'u'):
            r["human_label"] = "unsure"
        save_records(records)
    print("\n[INFO] Annotations saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run in interactive CLI mode")
    parser.add_argument("--port", type=int, default=5003, help="Port for web review UI")
    args = parser.parse_args()

    if args.cli:
        run_cli_mode()
    else:
        print(f"\n[INFO] Starting Phone Crop Review UI on http://127.0.0.1:{args.port}")
        print("[INFO] Open the URL in your browser to rapidly annotate crops.")
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
