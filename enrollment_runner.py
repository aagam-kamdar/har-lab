"""
enrollment_runner.py
--------------------
PHASE A: Enrollment Pipeline

Enrolls the 4 known students:
  - Aagam
  - Swayam
  - Harshil
  - Soham

Using:
  1. Primary face photos and body/appearance images from the face/ directory:
     face/aagam/
     face/swayam/
     face/harshil/
     face/soham/
  2. Opening camera-facing segment of new lab video.mp4 with ByteTrack to collect
     additional high-quality person observations.
  3. Builds persistent multi-crop body Re-ID gallery in reid/data/ with FAISS.

Zero cloud calls. Runs 100% locally.
"""

import os
import sys
import cv2
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from ultralytics import YOLO
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack
from reid.face_embedder import FaceEmbedder
from reid.enrolled_gallery import EnrolledGallery, MAX_BODY_EMBEDDINGS
from reid.embedding_model import PersonReIDExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enrollment")

# ========== CONFIGURATION ==========

FACE_DIR = "face"
STUDENT_NAMES = ["Aagam", "Swayam", "Harshil", "Soham"]
VIDEO_PATH = "new lab video.mp4"
YOLO_MODEL = "yolo26n.pt"
GALLERY_DATA_DIR = os.path.join("reid", "data")

ENROLLMENT_SECONDS = 30
MIN_BODY_CROPS_PER_STUDENT = 10


def run_enrollment():
    print("=" * 70)
    print("ENROLLMENT PIPELINE - Phase A")
    print("=" * 70)

    # 1. Verify face/ directory
    if not os.path.exists(FACE_DIR):
        print(f"ERROR: Face directory not found: {FACE_DIR}")
        sys.exit(1)

    for name in STUDENT_NAMES:
        folder = os.path.join(FACE_DIR, name.lower())
        if not os.path.exists(folder):
            print(f"ERROR: Missing student folder: {folder}")
            sys.exit(1)

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        sys.exit(1)

    print(f"\nStudents to enroll: {STUDENT_NAMES}")
    print(f"Reference images directory: {FACE_DIR}")
    print(f"Video: {VIDEO_PATH}")

    # 2. Initialize models & gallery
    print("\n[1/4] Loading models and initializing gallery...")
    face_embedder = FaceEmbedder()
    body_extractor = PersonReIDExtractor()
    
    # Fresh gallery in reid/data
    import shutil
    if os.path.exists(GALLERY_DATA_DIR):
        # Backup or clean
        pass
    gallery = EnrolledGallery(data_dir=GALLERY_DATA_DIR)
    # Reset FAISS & existing profiles to start clean
    gallery.faiss_index.reset()
    gallery.faiss_meta = []
    gallery.profiles = {}

    yolo = YOLO(YOLO_MODEL)
    print("  Models ready.")

    # 3. Ingest reference images from face/ directory
    print("\n[2/4] Ingesting student reference images from face/ directory...")
    for name in STUDENT_NAMES:
        folder = os.path.join(FACE_DIR, name.lower())
        files = sorted(os.listdir(folder))
        
        # Primary face photo
        primary_face = None
        for cand in [f"{name.lower()}.jpeg", f"{name.lower()}.jpg", f"{name.lower()}.png"]:
            if cand in files:
                primary_face = os.path.join(folder, cand)
                break
        if primary_face is None:
            primary_face = os.path.join(folder, files[0])

        gallery.enroll_face_photo(name, primary_face, face_embedder)
        print(f"  Enrolled face for {name} from {os.path.basename(primary_face)}")

        # Add all reference images (front, back, desk screenshots) to body gallery
        added_count = 0
        for f in files:
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path)
            if img is not None and img.shape[0] > 40 and img.shape[1] > 20:
                emb = body_extractor.extract(img)
                # force=True so curated reference images are always included
                if gallery.add_body_crop(name, img, emb, force=True):
                    added_count += 1
        print(f"  Added {added_count} curated body crops for {name}")

    # 4. Mine additional clean camera-facing observations from opening video
    print(f"\n[3/4] Mining video observations from opening segment (first {ENROLLMENT_SECONDS}s)...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frame = min(int(ENROLLMENT_SECONDS * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    BaseTrack.reset_id()
    tracker = BYTETracker(BYTETrackerArgs(track_thresh=0.35, track_buffer=60, match_thresh=0.8), frame_rate=int(round(fps)))

    frame_idx = 0
    video_added = {n: 0 for n in STUDENT_NAMES}

    while frame_idx < max_frame:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        if frame_idx % 2 != 0: continue  # every 2nd frame

        res = yolo(frame, conf=0.35, classes=[0], verbose=False)[0]
        dets = np.hstack([res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()[:,None]]) if len(res.boxes) > 0 else np.empty((0,5))
        targets = tracker.update(dets, (height, width), (height, width))

        for t in targets:
            x1, y1, x2, y2 = [int(v) for v in t.tlbr]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            w_box, h_box = x2 - x1, y2 - y1

            if w_box >= 60 and h_box >= 120:
                crop = frame[y1:y2, x1:x2]
                emb = body_extractor.extract(crop)

                # Search current gallery
                matches = gallery.search_body(emb, top_k=3)
                if matches:
                    top1 = matches[0]
                    top2 = None
                    for m in matches[1:]:
                        if m['student_name'] != top1['student_name']:
                            top2 = m
                            break
                    
                    # High confidence and distinct
                    margin = (top1['score'] - top2['score']) if top2 else 1.0
                    if top1['score'] >= 0.88 and margin >= 0.04:
                        sname = top1['student_name']
                        if len(gallery.profiles[sname].body_embeddings) < MAX_BODY_EMBEDDINGS:
                            if gallery.add_body_crop(sname, crop, emb, force=False):
                                video_added[sname] += 1

    cap.release()
    print(f"  Video mining added crops: {video_added}")

    # 5. Finalize & Save Gallery
    print("\n[4/4] Finalizing and saving gallery...")
    for name in STUDENT_NAMES:
        gallery.mark_enrollment_complete(name)

    gallery.save()

    print("\n" + "=" * 70)
    print("ENROLLMENT COMPLETE - Final Summary:")
    print("=" * 70)
    status = gallery.enrollment_status()
    for name, cnt in status.items():
        print(f"  ✓ {name:10s}: {cnt} total body embeddings in FAISS gallery")
    print(f"\nGallery directory: {GALLERY_DATA_DIR}")
    print(f"Total FAISS vectors: {gallery.total_body_embeddings()}")
    print("=" * 70)


if __name__ == "__main__":
    run_enrollment()
