# Hand-Crop Phone Detection Fine-Tuning Plan

**Document:** `FINE_TUNE_PLAN.md`  
**Status:** DRAFT / UNVERIFIED CANDIDATES (NO TRAINING RUN)  
**Target:** Computer Lab Productivity Monitor — Hand-Crop Phone & Call Verification  

> [!WARNING]
> **UNVERIFIED CANDIDATES AUDIT FINDING**: Visual inspection of 30 randomly sampled candidate frames revealed that 0/30 (0.0%) frames visibly contained a phone. The candidate frames were extracted via heuristic rule triggers (`wrist_phone < 1.10`) which fired on false detector proposals over empty desks, mouse pads, and typing hands. All 640 crops are classified as **unverified candidates** until human-confirmed ground truth is provided. No model training will occur without explicit human approval.  

---

## 1. Problem Diagnosis & Motivation
Extensive empirical evaluation on lab camera footage revealed:
1. **Severe Perception Failure for Phone Detection:**
   - Off-the-shelf COCO YOLOv8/26 achieves **0.0% full-frame recall** and only **7.5% wrist-crop recall** on ceiling-mounted cameras.
   - Smartphones appear small (25–45 px width) and are 60–80% occluded by fingers/palms.
2. **Call False Positives from Hand-on-Face Postures:**
   - 18.4% of candidate frames where students rest their chin or cheek on their hand trigger false-positive "call" state alerts due to geometric wrist-ear proximity ($< 0.60\text{ sw}$).
3. **Solution:**
   - Fine-tune a specialized lightweight detector on **$256 \times 256$ hand/wrist crops** to classify phone presence directly at the hand level.

---

## 2. Dataset Composition (`test_outputs/phone_dataset/`)
- **Total Images:** 640 curated $256 \times 256$ crops.
- **Positive Candidate Proposals (320 crops):**
  - Hands held in lap or lower chest regions during head-down postures.
  - Hands gripping small rectangular objects or potential handheld devices.
- **Hard Negative Crops (320 crops):**
  - `hard_neg_desk_keyboard_typing`: Hands resting flat on desks or typing on keyboards/laptops.
  - `hard_neg_hand_on_ear_chin`: Hands touching chin, temple, ear, or cheek (distinguishing thinking/resting from phone calls).
  - `hard_neg_gesturing_open_hand`: Open hands, pointing, or mid-air gesturing.
- **Annotation Template:** [`test_outputs/phone_dataset/annotations_template.csv`](file:///c:/Users/admin/OneDrive/Desktop/Projects/test_outputs/phone_dataset/annotations_template.csv)

---

## 3. Recommended Training Architecture & Regime
- **Base Architecture:** `yolov8n.pt` (transfer learning from COCO detection backbone).
- **Input Resolution:** $256 \times 256$ pixels (matches native crop scale without downsampling distortion).
- **Classes:** Single class (`phone`).
- **Optimization Strategy:**
  - **Phase 1 (Epochs 1–10):** Freeze backbone; train detection head using AdamW ($lr = 1\times 10^{-3}$, weight decay $5\times 10^{-4}$).
  - **Phase 2 (Epochs 11–50):** Unfreeze all layers; train with cosine learning rate schedule decaying to $1\times 10^{-5}$.
- **Data Augmentations:**
  - Mosaic augmentation & MixUp ($p=0.15$).
  - Random affine transformations: rotation $\pm 12^\circ$, shear $\pm 5^\circ$, scale $0.85 - 1.15$.
  - Photometric jitter: HSV saturation $\pm 0.4$, value $\pm 0.4$ (handles bright screens in dim rooms).
  - Synthetic finger/palm occlusion cutouts.

---

## 4. Production Inference Pipeline Integration
During real-time monitoring:
1. Full-frame YOLOv8-pose extracts 17 keypoints once per frame.
2. For each tracked person, crop $256 \times 256$ boxes centered on left wrist (`kp 9`) and right wrist (`kp 10`).
3. Batch all $2N$ wrist crops (e.g. 8 crops for 4 students) into a single forward pass:
   - Inference latency: $< 2.5\text{ms}$ on GPU / $\approx 10\text{ms}$ on CPU.
4. If phone confidence $> 0.40$ inside wrist crop:
   - Sets `wrist_phone_verified = True`.
   - Distinguishes genuine phone calls from thinking/chin-resting postures.
   - Eliminates the 18.4% false-positive call rate.

---

## 5. Governance & Safety Gate
> [!IMPORTANT]
> **MANDATORY HOLD:** In strict compliance with user instructions, **no training has been executed**.
> The dataset and manifest are ready for user inspection. Training will only begin once the user reviews the annotations and grants approval.
