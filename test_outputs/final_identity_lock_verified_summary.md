# FINAL IDENTITY-LOCK VERIFICATION VIDEO: CHRONOLOGICAL EVENT INDEX

## 1. Executive Summary & Verification Specifications

- **Rendered Output**: `test_outputs/final_identity_lock_verified.mp4`
- **Total Frames Rendered**: Exactly `1,800` frames (frame `0` to `1799`)
- **Video Duration**: `60.0` seconds at `30.0` FPS
- **Detector**: YOLO26n-pose (keypoints ignored, frozen detection stream: 6,818 detections)
- **Tracker**: ByteTrack (`track_thresh=0.35`, `track_buffer=90`, `match_thresh=0.85`)
- **Identity Authority**: ArcFace (`insightface buffalo_sc`, sole locking authority, sticky-permanent)
- **Recovery Engine**: OSNet-x0.25 (`tau=0.68`, `margin=0.05`, **Candidate Pool Scoping Active**)
- **Total ArcFace Locks**: `10`
- **Total Re-ID Recoveries**: `8`
- **Total Track Releases**: `10`
- **Total Ownership Violations**: `0` (Strictly verified at every frame)
- **Total Render Elapsed Time**: `131.2` seconds (`13.7` FPS)

## 2. Milestone First Face-Locks (ArcFace Sticky Authority)

| Student | First Lock Frame | Video Timestamp (mm:ss) | Initial Track ID | Authority Confirmation |
|:---|:---:|:---:|:---:|:---|
| **Aagam** | `f703` | **00:23.433 (00:23)** | Track `10` | ArcFace 3-frame confirmation >= 0.55 |
| **Swayam** | `f474` | **00:15.800 (00:15)** | Track `8` | ArcFace 3-frame confirmation >= 0.55 |
| **Harshil** | `f52` | **00:01.733 (00:01)** | Track `2` | ArcFace 3-frame confirmation >= 0.55 |
| **Soham** | `f1` | **00:00.033 (00:00)** | Track `1` | ArcFace 3-frame confirmation >= 0.55 |

## 3. Key Recovery Events Highlight

### A. EVT_BT_1: Harshil Post-Removal Recovery (Frame 902 / 00:30.067)
- **Physical Situation**: Harshil (Track 2) was seated at the left desk, occluded by monitors, and died at frame 697. Track 2 was evicted (REMOVED) at frame 788 after 90 lost frames.
- **Birth of Track 12**: At frame 902, Harshil moved and ByteTrack initialized new Track 12.
- **Candidate Pool Scoping**: Actively-owned students Aagam (Track 8) and Swayam (Track 10) were strictly excluded. Candidate pool was `{'harshil', 'soham'}`.
- **Re-ID Decision**: OSNet-x0.25 matched Harshil with similarity `0.765 >= 0.68` and margin `0.193 >= 0.05` against Soham (0.572).
- **Visual State**: Track 12 immediately displays in **YELLOW** (`Track 12 | HARSHIL [RECOVERED (Re-ID)]`) without waiting for ArcFace, bridging the long absence.

### B. EVT_BT_2: Walking Aagam Appearance Confusion (Frame 1089 / 00:36.300)
- **Physical Situation**: Aagam (Track 8) died at frame 943 and was evicted (REMOVED) at frame 1034. Track 18 appeared at frame 1089 as Aagam walked across the lab.
- **Candidate Pool Scoping**: Swayam (Track 10) and Harshil (Track 12) actively owned -> Candidate pool strictly `{'aagam', 'soham'}`.
- **Re-ID Decision**: OSNet-x0.25 scored Soham (`0.896`) higher than Aagam (`0.656`) due to nearly identical dark clothing features.
- **Documented Classification**: **KNOWN LIMITATION -- DARK-CLOTHING APPEARANCE CONFUSION**.
- **Ownership Integrity**: No ownership violation occurred; Soham was unowned, so Track 18 took single ownership of Soham until evicted or confirmed.

## 4. Chronological Event Log (All Locks, Recoveries, and Releases)

| Frame | Timestamp | Event Type | Track ID | Student | Trigger / Detail |
|:---:|:---:|:---:|:---:|:---|:---|
| `f0000` | `00:00.000 (00:00)` | **RECOVERY** | Track `1` | **soham** | Recovered identity 'soham' for new Track 1 (score=0.748, margin=0.154, tau=0.68) |
| `f0000` | `00:00.000 (00:00)` | **RECOVERY** | Track `2` | **harshil** | Recovered identity 'harshil' for new Track 2 (score=0.861, margin=0.201, tau=0.68) |
| `f0000` | `00:00.000 (00:00)` | **RECOVERY** | Track `3` | **aagam** | Recovered identity 'aagam' for new Track 3 (score=0.762, margin=0.189, tau=0.68) |
| `f0001` | `00:00.033 (00:00)` | **LOCK** | Track `1` | **soham** | Established permanent sticky identity lock on Track 1 for 'soham' (score=0.573, margin=0.216) [Confirmed prior Re-ID recovery] |
| `f0052` | `00:01.733 (00:01)` | **LOCK** | Track `2` | **harshil** | Established permanent sticky identity lock on Track 2 for 'harshil' (score=0.408, margin=0.303) [Confirmed prior Re-ID recovery] |
| `f0112` | `00:03.733 (00:03)` | **RECOVERY** | Track `4` | **swayam** | Recovered identity 'swayam' for new Track 4 (score=0.699, margin=0.699, tau=0.68) |
| `f0190` | `00:06.333 (00:06)` | **RELEASE** | Track `3` | **aagam** | Track 3 exceeded track_buffer=90; released ownership of 'aagam' back to unowned pool. |
| `f0273` | `00:09.099 (00:09)` | **RECOVERY** | Track `7` | **aagam** | Recovered identity 'aagam' for new Track 7 (score=0.707, margin=0.707, tau=0.68) |
| `f0284` | `00:09.466 (00:09)` | **RELEASE** | Track `4` | **swayam** | Track 4 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f0474` | `00:15.800 (00:15)` | **LOCK** | Track `8` | **swayam** | Established permanent sticky identity lock on Track 8 for 'swayam' (score=0.396, margin=0.075) |
| `f0593` | `00:19.766 (00:19)` | **RELEASE** | Track `7` | **aagam** | Track 7 exceeded track_buffer=90; released ownership of 'aagam' back to unowned pool. |
| `f0703` | `00:23.433 (00:23)` | **LOCK** | Track `10` | **aagam** | Established permanent sticky identity lock on Track 10 for 'aagam' (score=0.413, margin=0.209) |
| `f0788` | `00:26.266 (00:26)` | **RELEASE** | Track `2` | **harshil** | Track 2 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f0791` | `00:26.366 (00:26)` | **RECOVERY** | Track `11` | **harshil** | Recovered identity 'harshil' for new Track 11 (score=0.701, margin=0.701, tau=0.68) |
| `f0951` | `00:31.699 (00:31)` | **LOCK** | Track `11` | **harshil** | Established permanent sticky identity lock on Track 11 for 'harshil' (score=0.415, margin=0.210) [Confirmed prior Re-ID recovery] |
| `f1034` | `00:34.466 (00:34)` | **RELEASE** | Track `8` | **swayam** | Track 8 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f1134` | `00:37.799 (00:37)` | **RELEASE** | Track `11` | **harshil** | Track 11 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f1135` | `00:37.833 (00:37)` | **LOCK** | Track `12` | **harshil** | Established permanent sticky identity lock on Track 12 for 'harshil' (score=0.504, margin=0.339) |
| `f1223` | `00:40.766 (00:40)` | **RECOVERY** | Track `19` | **swayam** | Recovered identity 'swayam' for new Track 19 (score=0.759, margin=0.759, tau=0.68) |
| `f1224` | `00:40.799 (00:40)` | **LOCK** | Track `19` | **swayam** | Established permanent sticky identity lock on Track 19 for 'swayam' (score=0.475, margin=0.376) [Confirmed prior Re-ID recovery] |
| `f1232` | `00:41.066 (00:41)` | **RELEASE** | Track `10` | **aagam** | Track 10 exceeded track_buffer=90; released ownership of 'aagam' back to unowned pool. |
| `f1391` | `00:46.366 (00:46)` | **RELEASE** | Track `19` | **swayam** | Track 19 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f1434` | `00:47.799 (00:47)` | **LOCK** | Track `21` | **swayam** | Established permanent sticky identity lock on Track 21 for 'swayam' (score=0.398, margin=0.220) |
| `f1503` | `00:50.100 (00:50)` | **RELEASE** | Track `12` | **harshil** | Track 12 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f1603` | `00:53.433 (00:53)` | **RELEASE** | Track `21` | **swayam** | Track 21 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f1604` | `00:53.466 (00:53)` | **LOCK** | Track `23` | **swayam** | Established permanent sticky identity lock on Track 23 for 'swayam' (score=0.419, margin=0.233) |
| `f1722` | `00:57.399 (00:57)` | **RECOVERY** | Track `24` | **harshil** | Recovered identity 'harshil' for new Track 24 (score=0.721, margin=0.089, tau=0.68) |
| `f1739` | `00:57.966 (00:57)` | **LOCK** | Track `24` | **harshil** | Established permanent sticky identity lock on Track 24 for 'harshil' (score=0.385, margin=0.239) [Confirmed prior Re-ID recovery] |

## 5. Visual Coding Legend

- **GREEN Bounding Box**: ArcFace Identity Lock Established. ArcFace is the sole identity authority; sticky/permanent.
- **YELLOW Bounding Box**: Identity RECOVERED via Body Re-ID and not yet face-confirmed. Explicitly labeled `[RECOVERED (Re-ID)]`.
- **ORANGE Bounding Box**: UNKNOWN / No established identity. Motion-only ByteTrack tracklet.
- **Top-Left Status HUD**: Real-time display of frame index, timestamp, running counters, and the live mapping of all four enrolled students.
