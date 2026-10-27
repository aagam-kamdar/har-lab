# Computer Lab Activity Monitor: Recording Protocol & Scripted Scenarios

**Document:** `RECORDING_PROTOCOL.md`  
**Purpose:** Standard operating procedure for capturing clean, unambiguous, ground-truth video datasets for the computer lab productivity monitor.  
**Integrity Rule:** All newly recorded videos following this protocol must be placed in `sealed_test_set/` and strictly sequestered from model training, threshold tuning, and exploratory development until evaluation day.

---

## 1. Camera Setup & Recording Specifications

To ensure consistent detection across 30–40 students:
1. **Camera Angle & Mounting:**
   - Ceiling mount ($2.5\text{m} - 3.2\text{m}$ elevation), angled at $35^\circ - 45^\circ$ downward toward student workstations.
   - Dual-camera configuration:
     - `cam_0` (Main Lab View): Captures rows 1–3 and primary desk corridors.
     - `cam_1` (Secondary / Side View): Captures side profiles, resolving yaw occlusion and desk-edge blind spots.
2. **Resolution & Frame Rate:**
   - Native Resolution: **$1920 \times 1080$ (1080p)** or **$3840 \times 2160$ (4K)**.
   - Frame Rate: **$30.00$ FPS** constant frame rate (CFR). Avoid variable frame rate (VFR) containers.
   - Codec: H.264 / HEVC baseline MP4 with keyframe interval (GOP) $\le 30$ frames ($1.0\text{s}$) for millisecond-accurate seek.
3. **Lighting & Environment:**
   - Uniform fluorescent/LED lab overhead lighting (minimum 500 lux).
   - Eliminate direct window sun glare reflecting off monitor panels.

---

## 2. Scripted Activity Scenarios (Controlled Data Collection)

Each recording session should run for **3 to 5 minutes** per scenario with timestamps logged via a digital synchronizer clock visible in the frame corner.

### Scenario A: Phone Use & Interaction (`SCN_PHONE`)
*Objective: Provide ground-truth positives for phone detection in ceiling-view conditions.*
- **A1 — Lap Texting (Two Hands):** Student holds phone in lap with two thumbs typing, head tilted down ($30\text{s}$).
- **A2 — Chest-Height Browsing (One Hand):** Student holds phone at chest level, scrolling with thumb ($30\text{s}$).
- **A3 — Desk Flat Browsing:** Phone resting flat on desk adjacent to keyboard, index finger tapping/scrolling ($30\text{s}$).
- **A4 — Split Attention:** Left hand holding phone on desk/lap while right hand operates mouse/keyboard ($30\text{s}$).
- **A5 — Horizontal Media Viewing:** Phone held in landscape orientation with both hands ($30\text{s}$).

### Scenario B: Phone Calls (`SCN_CALL`)
*Objective: Validate call detection geometry (wrist at ear, elbow lifted away from desk).*
- **B1 — Right-Ear Call:** Hand holding phone to right ear, elbow lifted $>45^\circ$ above desk, talking into device ($45\text{s}$).
- **B2 — Left-Ear Call:** Hand holding phone to left ear, opposite arm resting on desk ($45\text{s}$).
- **B3 — Shoulder-Pinched Call:** Phone wedged between ear and shoulder while hands type on keyboard ($30\text{s}$).
- **B4 — Earbud / Speakerphone Interaction:** Phone held $10\text{cm}$ in front of mouth in speakerphone mode ($30\text{s}$).

### Scenario C: Desk Writing vs Typing (`SCN_WRITING`)
*Objective: Disambiguate handwriting from keyboard active work when keyboards share desk space.*
- **C1 — Paper Notebook Writing:** Student writes with pen in spiral notebook placed in front of keyboard, head tilted down ($60\text{s}$).
- **C2 — Tablet / Stylus Work:** Student writes with digital stylus on flat iPad/tablet ($45\text{s}$).
- **C3 — Textbook / Document Reading (No Writing):** Student seated steadily reading printed textbook with pen resting on desk ($45\text{s}$).
- **C4 — Alternating Typing & Note-Taking:** Student types for $10\text{s}$, writes for $10\text{s}$, cycles 3 times ($60\text{s}$).

### Scenario D: Peer Discussion & Collaboration (`SCN_DISCUSS`)
*Objective: Establish ground-truth boundary intervals for conversational collaboration.*
- **D1 — Direct Mutual Discussion:** Two adjacent students turn toward each other ($dx \approx 1.5\text{ sw}$), talking with hand gestures ($60\text{s}$).
- **D2 — Screen-Sharing Discussion:** Student A turns toward Student B's monitor, pointing at code on screen; Student B faces screen listening ($60\text{s}$).
- **D3 — Standing Consultation:** Student walks over to seated peer, stands at desk edge, both confer ($45\text{s}$).
- **D4 — Brief Inquiry (Micro-Turn):** Student leans over to ask a 3-second quick question, then resumes work ($15\text{s}$).

### Scenario E: Hard Negative Ambiguity Controls (`SCN_AMBIGUITY_NEG`)
*Objective: Eliminate false positives without suppressing true alerts.*
- **E1 — Chin / Jaw Resting (Thinking):** Student rests chin on palm with elbow propped on desk ($45\text{s}$) — *Must NOT trigger Call*.
- **E2 — Ear / Hair Scratching:** Student briefly scratches ear/head for $1.5\text{s}$ ($15\text{s}$) — *Must NOT trigger Call*.
- **E3 — Cheerful Slump / Head-Down Rest:** Student rests forehead on desk/forearms ($45\text{s}$) — *Must trigger Idle, NOT Phone*.
- **E4 — Sideways Glance / Staring into Space:** Student looks right at the wall/window without a neighbor present ($30\text{s}$) — *Must trigger Idle, NOT Discussion*.
- **E5 — Hand on Mouse / Palm Resting:** Hand resting motionless on mouse pad with curved fingers ($60\text{s}$) — *Must NOT trigger Phone*.

---

## 3. Ground-Truth Annotation Standards

When annotating intervals with the labeling tool:
1. **Source-Video Time:** All timestamps must be in presentation seconds ($t = \text{frame} / 30.0$).
2. **Interval Boundaries:**
   - `start_s`: The exact video timestamp where the physical motion/posture begins (e.g. wrist leaves desk and begins ascending toward ear).
   - `end_s`: The timestamp where the posture breaks (e.g. wrist moves away from ear).
3. **Exclusive Action Classes:**
   - `active_work`: Hands actively operating keyboard or mouse.
   - `writing`: Pen or stylus actively making marks on notebook or tablet on desk.
   - `reading`: Seated steadily reading monitor, paper, or book with minimal hand motion.
   - `discussing`: Verbal dialogue or shared attention with a peer.
   - `idle`: Seated at workstation without active work, discussion, or study.
   - `walking`: Upright standing or moving through the lab space.
   - `phone`: Active smartphone handling (lap, chest, or desk).
   - `call`: Phone handset held to ear with elbow lifted.
   - `away`: Student absent from desk/frame.
   - `uncertain`: Severe occlusion or ambiguous posture where human annotator cannot ascertain activity.
4. **Validation Hygiene:**
   - Minimum of two independent annotators per scenario recording.
   - Inter-annotator agreement (Cohen's Kappa $\kappa \ge 0.85$) required before inclusion into the sealed benchmark.
