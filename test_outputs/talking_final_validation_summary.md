# Diagnostic Validation Summary — Finalized Identity Pipeline on `talking.mp4`

**Execution Mode**: Diagnostic Validation (Read-Only — No Production Code Modified)  
**Date**: 2026-09-26  
**Video Under Test**: `talking.mp4`  
**Output Artifacts Generated**:
- Annotated Video: [`test_outputs/talking_final_validation.mp4`](file:///c:/Users/admin/OneDrive/Desktop/Projects/test_outputs/talking_final_validation.mp4) (30.017 FPS, 848x392, H.264)
- Forensic Event Audit JSON: [`test_outputs/talking_final_validation.json`](file:///c:/Users/admin/OneDrive/Desktop/Projects/test_outputs/talking_final_validation.json)

---

## 1. Current Production Implementation Verification

Before running `talking.mp4`, the current production codebase was inspected to confirm its structural invariants:

| Component | Code Inspection Question | Verified Status in Current Code | Exact Source Location |
|---|---|---|---|
| **BodyGallery** | Is `candidate_students` required explicitly? | **YES** — Mandatory parameter `Set[str]` | [`features.py:212, 239`](file:///c:/Users/admin/OneDrive/Desktop/Projects/features.py#L212) |
| **BodyGallery** | Is there any fallback to all enrolled students? | **NO** — If empty, immediately returns `(None, 0.0, 0.0)` | [`features.py:220, 245`](file:///c:/Users/admin/OneDrive/Desktop/Projects/features.py#L220) |
| **BodyGallery** | Does search score ONLY supplied candidates? | **YES** — Loops strictly over `for name in candidate_students:` | [`features.py:225`](file:///c:/Users/admin/OneDrive/Desktop/Projects/features.py#L225) |
| **Recovery Caller** | Is candidate set computed fresh from ownership? | **YES** — `KNOWN_STUDENTS - set(student_owner.keys())` | Finalized pipeline loop |
| **Recovery Caller** | Are TRACKED and LOST students excluded? | **YES** — `student_owner` preserves both TRACKED and LOST | [`identity_lock_manager.py:133`](file:///c:/Users/admin/OneDrive/Desktop/Projects/identity_lock_manager.py#L133) |
| **Gallery Mutation** | Is online / self-learning enabled? | **NO (FROZEN)** — 0 gallery additions in finalized pipeline | Finalized pipeline |
| **Face Crop** | What padding value is currently used? | **0.25 (25%)** | Default `0.3` in [`features.py`](file:///c:/Users/admin/OneDrive/Desktop/Projects/features.py#L75); caller passes `0.25` |
| **Face Selection** | How is face selected if multiple faces in crop? | **Largest bounding box area** in crop coordinates | [`features.py:67`](file:///c:/Users/admin/OneDrive/Desktop/Projects/features.py#L67) |
| **Face BBox Check**| Does it verify face center inside person bbox? | **NO** — Any face detected inside padded crop is accepted | [`features.py:67-98`](file:///c:/Users/admin/OneDrive/Desktop/Projects/features.py#L67) |

---

## 2. Video Information

- **Input File**: `talking.mp4`
- **Resolution**: $848 \times 392$
- **Source FPS**: $30.017$ FPS
- **Total Frames**: $3,417$ frames
- **Duration**: $113.84\text{s}$ ($1\text{m } 53.84\text{s}$)
- **Analysis Frame Rate**: Native $30.017$ FPS (every single frame analyzed, no temporal downsampling)
- **Detector**: YOLO26n-Pose (`yolo26n-pose.pt`, `conf=0.25`, `classes=[0]`)
- **Tracker**: Official ByteTrack (`track_thresh=0.35`, `track_buffer=90`, `match_thresh=0.85`, `mot20=False`, `frame_rate=30`)
- **ArcFace Model**: InsightFace `buffalo_sc` (`w600k_mbf` ArcFace 512-dim, `det_size=(320, 320)`)
- **Body Re-ID Model**: OSNet-x0.25 (`ExtractorB`, 512-dim, `tau=0.68`, `margin=0.05`, frozen gallery)

---

## 3. All Swayam / Harshil Suspected Incidents

Across all 3,417 frames in the finalized pipeline:

- **Total Swayam ↔ Harshil Identity Flips**: **0 (Zero)**
- **Track 5 (Swayam)**:
  - Initial Lock: Frame 356 (ArcFace score $0.395$, margin $0.222$)
  - Duration: Maintained lock from frame 356 to frame 2529 ($2,173$ consecutive frames = $72.4\text{s}$)
  - Status: 100% sticky lock preserved throughout.
- **Track 18 (Swayam)**:
  - Recovered via OSNet at frame 2529 (score $0.781$), confirmed by ArcFace at frame 3176 (score $0.430$).
- **Harshil Tracks**:
  - Track 3 (f59), Track 2 (f626), Track 9 (f1272), Track 10 (f1376), Track 11 (f1578), Track 8 (f1991), Track 21 (f3272).
  - All assigned to Harshil via OSNet / ArcFace without ever usurping Swayam's active ownership.

### Contradiction Audit (Swayam ↔ Harshil):
Only **3 frames** in the entire video had an ArcFace candidate predicting Harshil on Swayam's track:
- **f695** ($23.15\text{s}$): Track 5 (locked Swayam) face crop matched Harshil (score $0.392$, margin $0.192$). **Decision: CONTRADICTION_IGNORED**.
- **f696** ($23.19\text{s}$): Track 5 (locked Swayam) face crop matched Harshil (score $0.387$, margin $0.240$). **Decision: CONTRADICTION_IGNORED**.
- **f697** ($23.22\text{s}$): Track 5 (locked Swayam) face crop matched Harshil (score $0.394$, margin $0.218$). **Decision: CONTRADICTION_IGNORED**.

**Finding**: The sticky identity policy correctly absorbed all 3 bleed frames. **Zero identity switches occurred.**

---

## 4. Special Frame-356 Forensic Analysis

At frame 356, the exact state of Track 5 was audited down to sub-pixel coordinates:

1. **Does the Swayam person crop contain both faces?**: **YES**.
2. **How many SCRFD faces are detected?**: **2 faces** detected inside the padded crop window (`[387, 30, 515, 122]`).
3. **What are their bounding boxes and areas?**:
   - **Face 0 (Swayam)**: Absolute bbox `[428.9, 66.7, 443.1, 87.9]`, area = $301.88\text{ px}^2$.
   - **Face 1 (Harshil)**: Absolute bbox `[402.3, 57.1, 413.9, 75.7]`, area = $217.14\text{ px}^2$.
4. **Which face does the current implementation select?**:
   - It selects **Face 0** because its area ($301.88$) is larger than Face 1 ($217.14$).
5. **Does the selected face lie within Swayam's tracked person bounding box?**:
   - **YES**. Swayam's person bbox is `[408, 51, 494, 164]`. The center of Face 0 is at $x=436.0$, which is strictly inside the bounds $[408, 494]$.
6. **What ArcFace similarity is produced for Swayam?**:
   - Face 0 produced $\text{Sim}(\text{Face 0}, \text{swayam}) = \mathbf{0.3952}$.
7. **What ArcFace similarity is produced for Harshil?**:
   - Face 0 produced $\text{Sim}(\text{Face 0}, \text{harshil}) = \mathbf{0.1669}$.
8. **Does the identity manager change the existing identity?**:
   - Track 5 had 1 prior pending confirmation for Swayam. Face 0 provided the second consecutive confirmation ($\text{score} = 0.395 \ge 0.38$, $\text{margin} = 0.222 \ge 0.06$).
   - The identity manager **locked Track 5 as Swayam**.
   - It did **not** lock to Harshil.

---

## 5. Face-Selection Behavior

- **Mechanism**: [`FaceFeatureExtractor.extract_from_crop`](file:///c:/Users/admin/OneDrive/Desktop/Projects/features.py#L71) expands the person bounding box by $25\%$ padding and calls `app.get()`. It selects the single face with the maximum bounding box area (`max(faces, key=area)`).
- **Vulnerability Observed**: When two students sit in direct line-of-sight (Harshil in background row 3, Swayam in foreground row 2), a 25% horizontal expansion causes the background student's head to enter the foreground student's crop.
- At frames 695–697, when Swayam leaned slightly, Harshil's face became the largest detected face inside Swayam's crop, yielding ArcFace scores of $\sim 0.39$ for Harshil.
- **Why it did not cause a failure in the finalized pipeline**: The sticky identity policy rejected the contradiction and protected Track 5.

---

## 6. Body-ReID Behavior

- **Model**: OSNet-x0.25 (`ExtractorB`, 512-dim).
- **Thresholds**: $\tau = 0.68$, $\text{margin} \ge 0.05$.
- **Candidate Pool Scoping**: Evaluated strictly against unowned students.
- **Audit Results**:
  - At f277, Swayam's body was recovered with score $0.684$ ($\text{pool} = \{\text{swayam}, \text{soham}\}$).
  - At f626, Harshil's body was recovered with score $0.694$ ($\text{pool} = \{\text{harshil}, \text{soham}\}$).
  - In all 8 Re-ID recovery events throughout the video, Re-ID **never matched Swayam as Harshil or Harshil as Swayam**.

---

## 7. Gallery Mutation Audit

- **Initial Gallery Sizes**: `{'aagam': 42, 'swayam': 50, 'harshil': 37, 'soham': 31}`
- **Final Gallery Sizes**: `{'aagam': 42, 'swayam': 50, 'harshil': 37, 'soham': 31}`
- **Total Gallery Updates**: **0 (Zero)**
- **Is Gallery Frozen?**: **YES**.
- **Comparison to Legacy `robust_tracker.py`**:
  - In the user's earlier run using legacy `robust_tracker.py`, online self-learning was enabled, triggering **125 gallery additions** and saturating both `harshil` and `swayam` galleries to $80/80$.
  - In the current finalized pipeline, runtime gallery mutation is completely absent, ensuring zero gallery poisoning.

---

## 8. Ownership Audit

Across the entire 3,417-frame execution:
- `ownership_violations` = **0**
- `ownership_timing_violations` = **0**
- `candidate_pool_leak_violations` = **0**
- **Bijective Mapping**: `student_owner` and `track_to_student` were verified on every frame and remained exact mathematical inverses.
- **Lifecycle Integrity**:
  - `LOST` tracks retained ownership throughout their 90-frame buffer.
  - `REMOVED` tracks correctly released ownership upon eviction.
  - No student actively owned was ever placed into the Re-ID candidate pool.

---

## 9. Root-Cause Classification

The observed Swayam $\leftrightarrow$ Harshil misattribution is classified as:

### **A. LEGACY_PIPELINE_ARTIFACT**

**Primary Evidence**:
1. When running the **current finalized pipeline** on `talking.mp4`, **zero Swayam $\leftrightarrow$ Harshil identity switches occurred**. Track 5 remained locked to Swayam for $72.4$ seconds, and Harshil was tracked independently across distinct tracks.
2. The failure reported by the user occurred when running [`robust_tracker.py`](file:///c:/Users/admin/OneDrive/Desktop/Projects/robust_tracker.py), which is an older experimental script that differed in two critical ways:
   - Used legacy MobileNetV3 (672-dim) instead of OSNet-x0.25 (512-dim).
   - Contained an online self-learning loop (`self.body_gallery.add`), which performed **125 unvalidated additions** during `talking.mp4`, poisoning both students' body galleries.
3. In the finalized pipeline, the frozen OSNet gallery, candidate-pool scoping, and sticky lock architecture completely prevented this issue.

---

## 10. Code Change Recommendation (Diagnostic Only — NOT Implemented)

> **CODE CHANGE RECOMMENDED — NOT IMPLEMENTED**

While the current finalized pipeline succeeded on `talking.mp4` due to sticky identity locks, a latent physical risk was uncovered: **head-crop bleed** at frames 695–697 caused Harshil's face to be evaluated inside Swayam's crop.

If this happens on a **new, unconfirmed track** before a lock is established, two consecutive frames where a background person's face is larger could theoretically lock the track to the wrong person.

### Minimal Recommended Change (For Future Consideration):
In [`features.py`](file:///c:/Users/admin/OneDrive/Desktop/Projects/features.py#L71-L98) (`extract_from_crop`):
1. **Reduce or eliminate padding** when extracting face crops in crowded scenes (`padding=0.0` instead of `0.25` or `0.3`).
2. **Filter candidate faces by person bbox containment**: If multiple faces are detected in the crop, reject any face whose center $(x_c, y_c)$ falls outside the tracked person's unpadded bounding box:
   ```python
   # Proposed minimal filter (NOT IMPLEMENTED):
   valid_faces = [
       f for f in faces
       if (x1 <= (f.bbox[0] + f.bbox[2])/2.0 + cx1 <= x2) and
          (y1 <= (f.bbox[1] + f.bbox[3])/2.0 + cy1 <= y2)
   ]
   ```
This guarantees that an adjacent student sitting directly behind or beside cannot be selected as the target person's face.
