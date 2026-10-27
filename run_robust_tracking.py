"""
run_robust_tracking.py
----------------------
Robust Multi-Person Identity Tracking + Re-ID System on 'new lab video.mp4'.

Architecture:
  1. ENROLLMENT (Known Students: Aagam, Swayam, Harshil, Soham):
     - FRONT Photos → InsightFace SCRFD (detection) + ArcFace (512-dim face embeddings).
     - FRONT + BACK Photos → Realistic Augmentations (flips, rotations, jitter, blur, cutout)
       → PersonReIDExtractor (672-dim MobileNetV3 + HSV spatial color histograms)
       → Seed FAISS Vector Index.
     - Persisted to disk under 'reid/data/'.

  2. TRACKING PIPELINE:
     new lab video.mp4
          ↓
     Local YOLO Person-Only Detection (classes=[0])
          ↓
     Official BYTETracker (Sequential frame association)
          ↓
     Temporary track_id
          ↓
     Multi-Signal Identity Resolution:
       - If Face Detected (SCRFD + ArcFace) inside person bbox → Strong Face Identity Anchor
       - If Face Unavailable (desk/back view) → Body Re-ID via FAISS search
       - Multi-signal weighted fusion + Temporal Consensus Voting
          ↓
     Self-Learning / Adaptive Gallery (High-confidence body crops added to student gallery)
          ↓
     Unknown Protection (Enforces similarity threshold & ambiguity margin; No Student 05+)
          ↓
     Persistent Student ID (Aagam / Swayam / Harshil / Soham / UNKNOWN)
          ↓
     Output: test_outputs/robust_tracking.mp4
             (Label: Track ID | Name, e.g. 'Track 1 | Aagam')
"""

import os
import sys
import time
import json
import logging
import warnings
from collections import defaultdict, Counter

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from ultralytics import YOLO

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytetrack import BYTETracker, BYTETrackerArgs, BaseTrack, STrack, TrackState
from reid.embedding_model import PersonReIDExtractor
from reid.faiss_manager import FAISSManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Bounding box color palette locked to identity (BGR)
STUDENT_COLORS = {
    "Aagam": (0, 215, 255),     # Gold / Amber
    "Swayam": (255, 105, 180),  # Pink / Magenta
    "Harshil": (50, 205, 50),   # Bright Green
    "Soham": (255, 140, 0),     # Deep Orange
    "UNKNOWN": (140, 140, 140), # Gray
}


class UniversalVideoWriter:
    """Writes browser-playable MP4 video (H.264 via PyAV with cv2 fallback)."""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        self.fps = max(1.0, float(fps))
        self.width = width
        self.height = height
        self.mode = "av"
        self.container = None
        self.stream = None
        self.cv2_writer = None

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            import av
            self.container = av.open(output_path, mode="w")
            self.stream = self.container.add_stream("h264", rate=int(round(self.fps)))
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"preset": "veryfast", "crf": "23"}
            self.mode = "av"
            logger.info("Initialized PyAV H.264 video writer for: %s", output_path)
        except Exception as exc:
            logger.warning("PyAV failed (%s). Falling back to cv2.VideoWriter (mp4v).", exc)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.cv2_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            self.mode = "cv2"

    def write(self, bgr_frame: np.ndarray):
        if self.mode == "av":
            import av
            video_frame = av.VideoFrame.from_ndarray(bgr_frame, format="bgr24")
            for packet in self.stream.encode(video_frame):
                self.container.mux(packet)
        elif self.cv2_writer is not None:
            self.cv2_writer.write(bgr_frame)

    def release(self):
        if self.mode == "av" and self.container is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
            logger.info("Closed PyAV video writer.")
        elif self.cv2_writer is not None:
            self.cv2_writer.release()
            logger.info("Closed cv2.VideoWriter.")


# ---------------------------------------------------------------------------
# Realistic Body Augmentations (Requirement 3)
# ---------------------------------------------------------------------------
def generate_realistic_augmentations(crop: np.ndarray) -> list[np.ndarray]:
    """
    Generate realistic augmentations for body Re-ID without vertical flips or extreme distortion.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
        return []

    h, w = crop.shape[:2]
    augs = [crop.copy()]  # 1. Original

    # 2. Horizontal flip
    augs.append(cv2.flip(crop, 1))

    # 3. Small rotations (+-5 deg, +-10 deg)
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augs.append(rotated)

    # 4. Brightness / Contrast adjustments
    for alpha, beta in [(1.12, 12), (0.88, -12), (1.05, -15)]:
        adj = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        augs.append(adj)

    # 5. Saturation / Hue adjustments
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_mod = hsv.copy()
    hsv_mod[:, :, 1] = np.clip(hsv_mod[:, :, 1] * 1.15, 0, 255)
    augs.append(cv2.cvtColor(hsv_mod.astype(np.uint8), cv2.COLOR_HSV2BGR))

    # 6. Mild Gaussian blur
    augs.append(cv2.GaussianBlur(crop, (3, 3), 0.7))

    # 7. Mild noise
    noise = np.random.normal(0, 6, crop.shape).astype(np.float32)
    augs.append(np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8))

    # 8. Sharpening
    kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]], dtype=np.float32)
    augs.append(cv2.filter2D(crop, -1, kernel))

    # 9. Random small occlusion / cutout
    cutout = crop.copy()
    ch, cw = int(h * 0.12), int(w * 0.15)
    cy, cx = int(h * 0.35), int(w * 0.35)
    cutout[cy:cy + ch, cx:cx + cw] = 128
    augs.append(cutout)

    return augs


# ---------------------------------------------------------------------------
# Enrolled Identity Profiles (Face + Body)
# ---------------------------------------------------------------------------
class EnrolledIdentitySystem:
    """
    Manages Face profiles (ArcFace) + Body Re-ID galleries (FAISS) for the 4 students.
    """

    def __init__(
        self,
        face_app: FaceAnalysis,
        body_extractor: PersonReIDExtractor,
        data_dir: str = "reid/data",
    ):
        self.face_app = face_app
        self.body_extractor = body_extractor
        self.data_dir = data_dir
        self.faiss_mgr = FAISSManager(embedding_dim=body_extractor.embedding_dim)

        self.students = ["Aagam", "Swayam", "Harshil", "Soham"]
        self.face_profiles = {}     # student -> normalized 512-dim ArcFace embedding
        self.body_galleries = defaultdict(list) # student -> list of 672-dim embeddings
        self.gallery_counts = defaultdict(int)

        self.enroll()

    def enroll(self):
        """Enroll the 4 students using their front and back photos."""
        enrollment_files = {
            "Aagam": ("face/aagam/aagam.jpeg", "face/aagam/aagamback.jpg"),
            "Harshil": ("face/harshil/harshil.jpg", "face/harshil/harshilback.jpg"),
            "Soham": ("face/soham/soham.jpg", "face/soham/sohamback.png"),
            "Swayam": ("face/swayam/swayam.jpg", "face/swayam/swayamback.jpg"),
        }

        print("\n" + "=" * 70)
        print("ENROLLING 4 KNOWN STUDENTS (Face + Augmented Body Galleries)")
        print("=" * 70)

        for name in self.students:
            front_path, back_path = enrollment_files[name]
            f_img = cv2.imread(front_path)
            b_img = cv2.imread(back_path)

            if f_img is None:
                raise FileNotFoundError(f"Missing front image for {name}: {front_path}")

            # 1. Face Enrollment (Front Photo)
            faces = self.face_app.get(f_img)
            if faces:
                primary_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                f_emb = primary_face.embedding.astype(np.float32)
                f_emb = f_emb / np.linalg.norm(f_emb)
                self.face_profiles[name] = f_emb
                print(f"  [Face Enrolled] {name:<8s} | Score: {primary_face.det_score:.3f} | Bbox: {primary_face.bbox.astype(int)}")
            else:
                print(f"  [Warning] No face detected in front photo for {name}")

            # 2. Body Gallery Enrollment with Realistic Augmentations (Front + Back Photos)
            body_crops = [f_img]
            if b_img is not None:
                body_crops.append(b_img)

            added_for_student = 0
            for crop in body_crops:
                augmented_crops = generate_realistic_augmentations(crop)
                for aug in augmented_crops:
                    emb = self.body_extractor.extract(aug)
                    # Deduplicate: skip near-identical embeddings (>0.98 similarity to existing)
                    if any(float(np.dot(emb, existing)) > 0.98 for existing in self.body_galleries[name]):
                        continue

                    self.faiss_mgr.add_embedding(emb, name)
                    self.body_galleries[name].append(emb)
                    added_for_student += 1

            self.gallery_counts[name] = added_for_student
            print(f"  [Body Gallery]  {name:<8s} | Enrolled {added_for_student} diverse appearance vectors in FAISS")

        print(f"\nFAISS Total Appearance Vectors Indexed: {self.faiss_mgr.total_vectors}")

    def match_face(self, face_embedding: np.ndarray, threshold: float = 0.35, ambiguity_margin: float = 0.03):
        """Compare face against enrolled face profiles."""
        emb = face_embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm

        scores = {name: float(np.dot(emb, profile)) for name, profile in self.face_profiles.items()}
        sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_name, best_score = sorted_cands[0]
        second_name, second_score = sorted_cands[1]

        is_ambiguous = (best_score - second_score) < ambiguity_margin
        if best_score >= threshold and not is_ambiguous:
            return best_name, best_score
        return "UNKNOWN", best_score

    def match_body(self, body_embedding: np.ndarray, threshold: float = 0.70, ambiguity_margin: float = 0.03):
        """Query FAISS index for best body appearance match."""
        results = self.faiss_mgr.search(body_embedding, top_k=5)
        if not results:
            return "UNKNOWN", 0.0

        top_match = results[0]
        top_name = top_match["student_id"]
        top_score = top_match["score"]

        # Check ambiguity
        is_ambiguous = False
        for other in results[1:]:
            if other["student_id"] != top_name:
                if (top_score - other["score"]) < ambiguity_margin:
                    is_ambiguous = True
                break

        if top_score >= threshold and not is_ambiguous:
            return top_name, top_score
        return "UNKNOWN", top_score

    def update_gallery(self, student_name: str, body_crop: np.ndarray, max_gallery_size: int = 35) -> bool:
        """Adaptive Gallery: add high-confidence new observation to student's gallery."""
        if student_name not in self.students:
            return False

        if len(self.body_galleries[student_name]) >= max_gallery_size:
            return False

        emb = self.body_extractor.extract(body_crop)
        # Avoid duplicate vectors
        if any(float(np.dot(emb, existing)) > 0.96 for existing in self.body_galleries[student_name]):
            return False

        self.faiss_mgr.add_embedding(emb, student_name)
        self.body_galleries[student_name].append(emb)
        self.gallery_counts[student_name] += 1
        return True


# ---------------------------------------------------------------------------
# Robust Multi-Signal Identity Resolver
# ---------------------------------------------------------------------------
class RobustIdentityResolver:
    """
    Connects temporary ByteTrack track IDs to persistent Student identities.
    Combines Face + Body Re-ID + Temporal Voting + Unknown Protection.
    """

    def __init__(
        self,
        enrolled_sys: EnrolledIdentitySystem,
        min_consensus: int = 3,
        fps: float = 30.0,
    ):
        self.sys = enrolled_sys
        self.min_consensus = min_consensus
        self.fps = fps

        # Track ID -> Persistent Student Name
        self.track_to_student = {}

        # Student -> List of Track IDs
        self.student_to_tracks = defaultdict(list)

        # Temporal observation voting buffer: track_id -> list of candidate names
        self.observation_buffer = defaultdict(list)

        # Last seen info: student -> (last_frame, last_track_id)
        self.student_last_seen = {}

        # Metrics accumulators
        self.total_face_matches = 0
        self.total_body_matches = 0
        self.identity_recoveries = 0
        self.identity_switches = 0
        self.unknown_periods = 0
        self.gallery_updates = 0
        self.all_similarity_scores = []
        self.recovery_log = []

    def get_identity(self, track_id: int) -> str:
        return self.track_to_student.get(track_id, "UNKNOWN")

    def resolve_track(
        self,
        track_id: int,
        person_crop: np.ndarray,
        detected_faces_in_frame: list,
        person_bbox: tuple,
        frame_idx: int,
    ) -> str:
        """
        Determine or confirm the identity of a track using multi-signal evidence.
        """
        # If already confirmed, maintain identity
        if track_id in self.track_to_student:
            student_id = self.track_to_student[track_id]
            self.student_last_seen[student_id] = (frame_idx, track_id)

            # Optional self-learning on high-quality frames
            if frame_idx % 60 == 0:
                if self.sys.update_gallery(student_id, person_crop):
                    self.gallery_updates += 1

            return student_id

        if person_crop is None or person_crop.size == 0:
            return "UNKNOWN"

        px1, py1, px2, py2 = person_bbox

        # 1. Check if a face is inside this person's bounding box
        face_match_name = None
        face_score = 0.0

        for f in detected_faces_in_frame:
            fx1, fy1, fx2, fy2 = f.bbox
            # Center of face inside person bbox
            fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            if px1 <= fcx <= px2 and py1 <= fcy <= py2:
                name, score = self.sys.match_face(f.embedding, threshold=0.38)
                if name != "UNKNOWN":
                    face_match_name = name
                    face_score = score
                    self.total_face_matches += 1
                    self.all_similarity_scores.append(score)
                    break

        # 2. Body Re-ID matching
        body_emb = self.sys.body_extractor.extract(person_crop)
        body_match_name, body_score = self.sys.match_body(body_emb, threshold=0.72)
        if body_match_name != "UNKNOWN":
            self.total_body_matches += 1
            self.all_similarity_scores.append(body_score)

        # 3. Decision & Multi-Signal Fusion
        if face_match_name:
            chosen_candidate = face_match_name
            chosen_score = face_score
            signal_type = "Face"
        elif body_match_name != "UNKNOWN":
            chosen_candidate = body_match_name
            chosen_score = body_score
            signal_type = "Body"
        else:
            chosen_candidate = "UNKNOWN"
            chosen_score = max(face_score, body_score)
            signal_type = "None"

        # 4. Temporal Consensus Buffer (Requirement 1 & 7)
        self.observation_buffer[track_id].append(chosen_candidate)
        obs = self.observation_buffer[track_id]

        if len(obs) >= self.min_consensus:
            valid_cands = [c for c in obs if c != "UNKNOWN"]
            if valid_cands:
                top_cand, count = Counter(valid_cands).most_common(1)[0]
                if count >= (self.min_consensus // 2 + 1):
                    # CONFIRMED
                    self.track_to_student[track_id] = top_cand
                    self.student_to_tracks[top_cand].append(track_id)

                    # Check if this was a long-term recovery
                    if top_cand in self.student_last_seen:
                        old_last_frame, old_tid = self.student_last_seen[top_cand]
                        if old_tid != track_id:
                            gap = frame_idx - old_last_frame
                            gap_s = gap / self.fps
                            self.identity_recoveries += 1
                            self.recovery_log.append({
                                "old_track": old_tid,
                                "new_track": track_id,
                                "student": top_cand,
                                "similarity": float(chosen_score),
                                "gap_frames": gap,
                                "gap_sec": gap_s,
                                "signal": signal_type,
                            })
                            logger.info(
                                "IDENTITY RECOVERED: Track %d -> %s (previously Track %d, gap=%.1fs, signal=%s)",
                                track_id, top_cand, old_tid, gap_s, signal_type
                            )

                    self.student_last_seen[top_cand] = (frame_idx, track_id)

                    # Update gallery with confirmed video observation (Requirement 4)
                    if self.sys.update_gallery(top_cand, person_crop):
                        self.gallery_updates += 1

                    return top_cand

        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Annotation & Visualization (Requirement 9)
# ---------------------------------------------------------------------------
def draw_robust_annotations(
    frame: np.ndarray,
    tracks: list[STrack],
    resolver: RobustIdentityResolver,
) -> np.ndarray:
    """
    Draw bounding boxes and banners: 'Track ID | Name' (e.g. 'Track 1 | Aagam').
    Colors are locked to the persistent student identity.
    """
    h_fr, w_fr = frame.shape[:2]

    for trk in tracks:
        tid = trk.track_id
        student_name = resolver.get_identity(tid)

        x1, y1, x2, y2 = [int(v) for v in trk.tlbr]
        x1, y1 = max(0, min(x1, w_fr - 1)), max(0, min(y1, h_fr - 1))
        x2, y2 = max(0, min(x2, w_fr - 1)), max(0, min(y2, h_fr - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = STUDENT_COLORS.get(student_name, (140, 140, 140))

        # Person bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Banner label: "Track X | Name"
        label = f"Track {tid} | {student_name}"
        font_scale = 0.55
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        label_top = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, label_top), (x1 + tw + 8, y1), color, -1)

        text_color = (0, 0, 0) if student_name != "UNKNOWN" else (255, 255, 255)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return frame


# ---------------------------------------------------------------------------
# Main Pipeline Runner (Requirement 6, 9, 10)
# ---------------------------------------------------------------------------
def run_robust_tracking_pipeline(
    video_path="new lab video.mp4",
    output_video_path="test_outputs/robust_tracking.mp4",
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("\n" + "=" * 75)
    print("INITIALIZING ROBUST PERSON TRACKING & IDENTITY RE-ID SYSTEM")
    print("=" * 75)

    # 1. Initialize Face Analysis (SCRFD + ArcFace)
    print("Loading Face Analysis models (SCRFD + ArcFace)...")
    face_app = FaceAnalysis(
        name="buffalo_sc",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(480, 480))

    # 2. Initialize Body Re-ID Feature Extractor (MobileNetV3 + HSV spatial)
    print("Loading Body Re-ID Appearance Extractor...")
    body_extractor = PersonReIDExtractor()

    # 3. Enroll 4 Known Identities (Front Face + Augmented Body Galleries)
    enrolled_sys = EnrolledIdentitySystem(face_app, body_extractor)

    # 4. Initialize Local YOLO Person-Only Detector
    print("Loading Local YOLO Person Detector (classes=[0])...")
    yolo = YOLO("yolo26n.pt")

    # 5. Initialize Official BYTETracker
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    BaseTrack.reset_id()
    args = BYTETrackerArgs(
        track_thresh=0.50,
        track_buffer=30,
        match_thresh=0.80,
        mot20=False,
    )
    tracker = BYTETracker(args=args, frame_rate=int(round(fps)))

    # 6. Initialize Identity Resolver
    resolver = RobustIdentityResolver(enrolled_sys, min_consensus=3, fps=fps)

    # 7. Initialize Output Video Writer
    writer = UniversalVideoWriter(output_video_path, fps, width, height)

    print("\n" + "=" * 75)
    print("RUNNING ROBUST TRACKING ON 'new lab video.mp4'")
    print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Frames: {total_frames} ({total_frames/fps:.1f}s)")
    print(f"Output Video: {output_video_path}")
    print("=" * 75 + "\n")

    # Frame tracking stats
    per_student_frames = defaultdict(int)
    all_unique_tracks = set()
    total_person_detections = 0
    start_time = time.time()

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # A. YOLO Person Detection (classes=[0] -> Person only)
        yolo_res = yolo(frame, conf=0.30, classes=[0], verbose=False)[0]
        num_dets = len(yolo_res.boxes)
        total_person_detections += num_dets

        if num_dets > 0:
            xyxy = yolo_res.boxes.xyxy.cpu().numpy()
            conf = yolo_res.boxes.conf.cpu().numpy()[:, None]
            dets = np.hstack([xyxy, conf])
        else:
            dets = np.empty((0, 5), dtype=float)

        # B. SCRFD Face Detection for this frame
        faces = face_app.get(frame)

        # C. Official BYTETracker Update
        online_targets = tracker.update(dets, (height, width), (height, width))

        active_in_frame = set()

        for trk in online_targets:
            tid = trk.track_id
            all_unique_tracks.add(tid)

            x1, y1, x2, y2 = [int(v) for v in trk.tlbr]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            crop = frame[y1:y2, x1:x2] if (x2 > x1 + 10 and y2 > y1 + 10) else None

            # D. Multi-Signal Identity Resolution
            student_id = resolver.resolve_track(
                track_id=tid,
                person_crop=crop,
                detected_faces_in_frame=faces,
                person_bbox=(x1, y1, x2, y2),
                frame_idx=frame_idx,
            )

            if student_id != "UNKNOWN":
                per_student_frames[student_id] += 1
                active_in_frame.add(student_id)

        # Track unknown periods
        if len(online_targets) > len(active_in_frame):
            resolver.unknown_periods += 1

        # E. Annotate & Write Video
        annotated_frame = draw_robust_annotations(frame, online_targets, resolver)
        writer.write(annotated_frame)

        if frame_idx % 300 == 0 or frame_idx == 1:
            elapsed = time.time() - start_time
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            active_str = ", ".join([f"T{t.track_id}:{resolver.get_identity(t.track_id)}" for t in online_targets])
            logger.info(
                "Frame %4d / %d (%.1fs) | Speed: %.1f FPS | Tracks: %s",
                frame_idx, total_frames, frame_idx / fps, proc_fps, active_str or "None"
            )

    cap.release()
    writer.release()

    total_proc_time = time.time() - start_time

    # -------------------------------------------------------------
    # GENERATE COMPREHENSIVE EVALUATION REPORT (Requirement 10)
    # -------------------------------------------------------------
    SEP = "=" * 80
    print("\n" + SEP)
    print("ROBUST PERSON TRACKING & IDENTITY RE-ID EVALUATION REPORT")
    print(SEP)
    print(f"Video File Processed               : {video_path}")
    print(f"Total Video Frames                 : {frame_idx}")
    print(f"Video Duration                     : {frame_idx/fps:.2f} seconds ({frame_idx/fps/60:.2f} mins)")
    print(f"Processing Time                    : {total_proc_time:.1f}s ({frame_idx/total_proc_time:.1f} FPS)")
    print(f"Total YOLO Person Detections       : {total_person_detections}")
    print(f"Total Temporary ByteTrack IDs      : {len(all_unique_tracks)}")
    print(f"Persistent Enrolled Identities     : {', '.join(enrolled_sys.students)}")
    print(f"Successful Identity Recoveries     : {resolver.identity_recoveries}")
    print(f"Identity Switches                  : {resolver.identity_switches}")
    print(f"Frames with UNKNOWN Track Periods  : {resolver.unknown_periods} ({100*resolver.unknown_periods/frame_idx:.1f}%)")
    print(f"Total Face-Assisted Matches        : {resolver.total_face_matches}")
    print(f"Total Body-ReID Matches            : {resolver.total_body_matches}")
    print(f"Adaptive Gallery Embeddings Added  : {resolver.gallery_updates}")

    sims = resolver.all_similarity_scores
    if sims:
        print(f"Similarity Scores (Min/Avg/Max)    : {min(sims):.3f} / {np.mean(sims):.3f} / {max(sims):.3f}")

    print("\n" + SEP)
    print("PER-STUDENT IDENTITY COVERAGE")
    print(SEP)
    print(f"{'Student Name':<12} | {'Active Frames':>13} | {'Coverage (%)':>12} | {'Associated Track IDs':<30}")
    print("-" * 75)
    for name in enrolled_sys.students:
        f_cnt = per_student_frames.get(name, 0)
        cov = 100.0 * f_cnt / frame_idx
        tids = resolver.student_to_tracks.get(name, [])
        tids_str = ", ".join([f"Track {t}" for t in tids]) if tids else "None"
        print(f"{name:<12} | {f_cnt:>13d} | {cov:>11.1f}% | {tids_str:<30}")

    print("\n" + SEP)
    print("FINAL TEMPORARY TRACK ID -> PERSISTENT STUDENT MAPPING")
    print(SEP)
    print(f"{'Track ID':>8} | {'Persistent Student Identity':<28} | State")
    print("-" * 55)
    for tid in sorted(all_unique_tracks):
        assigned = resolver.track_to_student.get(tid, "UNKNOWN")
        state = "Confirmed Identity" if assigned != "UNKNOWN" else "Unconfirmed / Unknown"
        print(f"Track {tid:2d} | {assigned:<28} | {state}")

    if resolver.recovery_log:
        print("\n" + SEP)
        print("SAMPLE IDENTITY RECOVERY TRANSITIONS (Across Gaps & Occlusions)")
        print(SEP)
        print(f"{'Old Track':>9} | {'New Track':>9} | {'Student Name':<12} | {'Similarity':>10} | {'Gap (Frames)':>12} | {'Signal':<8}")
        print("-" * 70)
        for ev in resolver.recovery_log[:15]:
            print(f"Track {ev['old_track']:4d} | Track {ev['new_track']:4d} | {ev['student']:<12} | {ev['similarity']:10.3f} | {ev['gap_frames']:12d} | {ev['signal']:<8}")

    print("\n" + SEP)
    print(f"Annotated Robust Tracking Video Generated: {output_video_path}")
    print(f"File Size: {os.path.getsize(output_video_path):,} bytes")
    print(SEP)


if __name__ == "__main__":
    run_robust_tracking_pipeline(
        video_path="new lab video.mp4",
        output_video_path="test_outputs/robust_tracking.mp4",
    )
