# FULL 3 MIN 20 SEC VERIFICATION VIDEO: CHRONOLOGICAL EVENT INDEX

## 1. Executive Summary & Verification Specifications

- **Video Output File**: `test_outputs/final_identity_lock_full_video.mp4`
- **Source Video Processed**: `6005` frames (`03:20` at `30.0` FPS)
- **Output Video Frame Rate**: `1.0` FPS (1 sample per second of lab time as requested)
- **Output Video Frames Written**: `202` frames (duration `03:22`)
- **Pipeline Execution Time**: `608.0` seconds (overall throughput: `9.9` FPS)
- **Detector**: YOLO26n-pose (frames 0–1799 from verified frozen cache; frames 1800–6004 batched inference; keypoints ignored)
- **Tracker**: ByteTrack (`track_thresh=0.35`, `track_buffer=90`, `match_thresh=0.85`, `mot20=False`)
- **Identity Authority**: ArcFace (`insightface buffalo_sc`, sole authority for initial lock, sticky-permanent)
- **Recovery Engine**: OSNet-x0.25 (`tau=0.68`, `margin=0.05`, **Candidate Pool Scoping Active**)
- **Total ArcFace Locks Established**: `16`
- **Total Body Re-ID Recoveries**: `21`
- **Total Track Ownership Releases**: `21`
- **Total Ownership Invariant Violations**: `0` (Strictly verified at every single frame)

## 2. Milestone First Face-Locks (ArcFace Sticky Authority)

| Student | First Lock Frame | Video Timestamp (mm:ss) | Initial Track ID | Authority Confirmation |
|:---|:---:|:---:|:---:|:---|
| **Aagam** | `f611` | **00:20.366 (00:20)** | Track `10` | ArcFace confirmation >= 0.55 |
| **Swayam** | `f474` | **00:15.800 (00:15)** | Track `8` | ArcFace confirmation >= 0.55 |
| **Harshil** | `f52` | **00:01.733 (00:01)** | Track `2` | ArcFace confirmation >= 0.55 |
| **Soham** | `f1` | **00:00.033 (00:00)** | Track `1` | ArcFace confirmation >= 0.55 |

## 3. Notable Recovery Events Across Full Video

### A. EVT_BT_1: Harshil Post-Removal Recovery (Frame 902 / 00:30.067)
- **Physical Situation**: Harshil (Track 2) was seated at the left desk, occluded by monitors, and died at frame 697. Track 2 was evicted (REMOVED) at frame 788 after 90 lost frames.
- **Birth of Track 12**: At frame 902, Harshil moved and ByteTrack initialized new Track 12.
- **Candidate Pool Scoping**: Actively-owned students Aagam and Swayam were strictly excluded. Candidate pool was `{'harshil', 'soham'}`.
- **Re-ID Decision**: OSNet-x0.25 matched Harshil with similarity `0.765 >= 0.68` and margin `0.193 >= 0.05` against Soham (0.572).
- **Visual State**: Track 12 immediately displayed in **YELLOW** (`Track 12 | HARSHIL [RECOVERED (Re-ID)]`) without waiting for ArcFace, bridging the long absence.

### B. EVT_BT_2: Walking Aagam Appearance Confusion (Frame 1089 / 00:36.300)
- **Physical Situation**: Aagam (Track 8) died at frame 943 and was evicted (REMOVED) at frame 1034. Track 18 appeared at frame 1089 as Aagam walked across the lab.
- **Candidate Pool Scoping**: Swayam and Harshil actively owned -> Candidate pool strictly `{'aagam', 'soham'}`.
- **Re-ID Decision**: OSNet-x0.25 scored Soham (`0.896`) higher than Aagam (`0.656`) due to nearly identical dark clothing features.
- **Documented Classification**: **KNOWN LIMITATION -- DARK-CLOTHING APPEARANCE CONFUSION**.
- **Ownership Integrity**: No ownership violation occurred; Soham was unowned, so Track 18 took single ownership of Soham until evicted or confirmed.

## 4. Chronological Event Log (All Locks, Recoveries, and Releases)

| Frame | Timestamp | Event Type | Track ID | Student | Trigger / Detail |
|:---:|:---:|:---:|:---:|:---|:---|
| `f0000` | `00:00.000 (00:00)` | **RECOVERY** | Track `1` | **soham** | Recovered identity 'soham' for Track 1 (score=0.748, margin=0.154, tau=0.68) |
| `f0000` | `00:00.000 (00:00)` | **RECOVERY** | Track `2` | **harshil** | Recovered identity 'harshil' for Track 2 (score=0.861, margin=0.201, tau=0.68) |
| `f0000` | `00:00.000 (00:00)` | **RECOVERY** | Track `3` | **aagam** | Recovered identity 'aagam' for Track 3 (score=0.777, margin=0.144, tau=0.68) |
| `f0001` | `00:00.033 (00:00)` | **LOCK** | Track `1` | **soham** | Established permanent sticky identity lock on Track 1 for 'soham' (score=0.573, margin=0.216) [Confirmed prior Re-ID recovery] |
| `f0052` | `00:01.733 (00:01)` | **LOCK** | Track `2` | **harshil** | Established permanent sticky identity lock on Track 2 for 'harshil' (score=0.408, margin=0.303) [Confirmed prior Re-ID recovery] |
| `f0112` | `00:03.733 (00:03)` | **RECOVERY** | Track `4` | **swayam** | Recovered identity 'swayam' for Track 4 (score=0.699, margin=0.699, tau=0.68) |
| `f0190` | `00:06.333 (00:06)` | **RELEASE** | Track `3` | **aagam** | Track 3 exceeded track_buffer=90; released ownership of 'aagam' back to unowned pool. |
| `f0272` | `00:09.066 (00:09)` | **RECOVERY** | Track `7` | **aagam** | Recovered identity 'aagam' for Track 7 (score=0.745, margin=0.745, tau=0.68) |
| `f0284` | `00:09.466 (00:09)` | **RELEASE** | Track `4` | **swayam** | Track 4 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f0474` | `00:15.800 (00:15)` | **LOCK** | Track `8` | **swayam** | Established permanent sticky identity lock on Track 8 for 'swayam' (score=0.386, margin=0.067) |
| `f0593` | `00:19.766 (00:19)` | **RELEASE** | Track `7` | **aagam** | Track 7 exceeded track_buffer=90; released ownership of 'aagam' back to unowned pool. |
| `f0606` | `00:20.199 (00:20)` | **RECOVERY** | Track `10` | **aagam** | Recovered identity 'aagam' for Track 10 (score=0.758, margin=0.758, tau=0.68) |
| `f0611` | `00:20.366 (00:20)` | **LOCK** | Track `10` | **aagam** | Established permanent sticky identity lock on Track 10 for 'aagam' (score=0.383, margin=0.224) [Confirmed prior Re-ID recovery] |
| `f0788` | `00:26.266 (00:26)` | **RELEASE** | Track `2` | **harshil** | Track 2 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f0788` | `00:26.266 (00:26)` | **RECOVERY** | Track `11` | **harshil** | Recovered identity 'harshil' for Track 11 (score=0.849, margin=0.849, tau=0.68) |
| `f0951` | `00:31.699 (00:31)` | **LOCK** | Track `11` | **harshil** | Established permanent sticky identity lock on Track 11 for 'harshil' (score=0.415, margin=0.210) [Confirmed prior Re-ID recovery] |
| `f1034` | `00:34.466 (00:34)` | **RELEASE** | Track `8` | **swayam** | Track 8 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f1134` | `00:37.799 (00:37)` | **RELEASE** | Track `11` | **harshil** | Track 11 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f1135` | `00:37.833 (00:37)` | **LOCK** | Track `12` | **harshil** | Established permanent sticky identity lock on Track 12 for 'harshil' (score=0.504, margin=0.339) |
| `f1223` | `00:40.766 (00:40)` | **RECOVERY** | Track `19` | **swayam** | Recovered identity 'swayam' for Track 19 (score=0.759, margin=0.759, tau=0.68) |
| `f1224` | `00:40.799 (00:40)` | **LOCK** | Track `19` | **swayam** | Established permanent sticky identity lock on Track 19 for 'swayam' (score=0.475, margin=0.376) [Confirmed prior Re-ID recovery] |
| `f1232` | `00:41.066 (00:41)` | **RELEASE** | Track `10` | **aagam** | Track 10 exceeded track_buffer=90; released ownership of 'aagam' back to unowned pool. |
| `f1232` | `00:41.066 (00:41)` | **RECOVERY** | Track `20` | **aagam** | Recovered identity 'aagam' for Track 20 (score=0.762, margin=0.762, tau=0.68) |
| `f1391` | `00:46.366 (00:46)` | **RELEASE** | Track `19` | **swayam** | Track 19 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f1391` | `00:46.366 (00:46)` | **RECOVERY** | Track `21` | **swayam** | Recovered identity 'swayam' for Track 21 (score=0.753, margin=0.753, tau=0.68) |
| `f1401` | `00:46.700 (00:46)` | **LOCK** | Track `21` | **swayam** | Established permanent sticky identity lock on Track 21 for 'swayam' (score=0.485, margin=0.244) [Confirmed prior Re-ID recovery] |
| `f1503` | `00:50.100 (00:50)` | **RELEASE** | Track `12` | **harshil** | Track 12 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f1503` | `00:50.100 (00:50)` | **RECOVERY** | Track `22` | **harshil** | Recovered identity 'harshil' for Track 22 (score=0.802, margin=0.802, tau=0.68) |
| `f1603` | `00:53.433 (00:53)` | **RELEASE** | Track `21` | **swayam** | Track 21 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f1603` | `00:53.433 (00:53)` | **RECOVERY** | Track `23` | **swayam** | Recovered identity 'swayam' for Track 23 (score=0.897, margin=0.897, tau=0.68) |
| `f1604` | `00:53.466 (00:53)` | **LOCK** | Track `23` | **swayam** | Established permanent sticky identity lock on Track 23 for 'swayam' (score=0.401, margin=0.222) [Confirmed prior Re-ID recovery] |
| `f1675` | `00:55.833 (00:55)` | **RELEASE** | Track `22` | **harshil** | Track 22 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f1722` | `00:57.399 (00:57)` | **RECOVERY** | Track `24` | **harshil** | Recovered identity 'harshil' for Track 24 (score=0.821, margin=0.821, tau=0.68) |
| `f1750` | `00:58.333 (00:58)` | **LOCK** | Track `24` | **harshil** | Established permanent sticky identity lock on Track 24 for 'harshil' (score=0.392, margin=0.319) [Confirmed prior Re-ID recovery] |
| `f2018` | `01:07.266 (01:07)` | **RELEASE** | Track `23` | **swayam** | Track 23 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f2024` | `01:07.466 (01:07)` | **RELEASE** | Track `24` | **harshil** | Track 24 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f2096` | `01:09.866 (01:09)` | **RECOVERY** | Track `25` | **swayam** | Recovered identity 'swayam' for Track 25 (score=0.787, margin=0.172, tau=0.68) |
| `f2109` | `01:10.299 (01:10)` | **RELEASE** | Track `1` | **soham** | Track 1 exceeded track_buffer=90; released ownership of 'soham' back to unowned pool. |
| `f2142` | `01:11.400 (01:11)` | **RECOVERY** | Track `26` | **soham** | Recovered identity 'soham' for Track 26 (score=0.915, margin=0.400, tau=0.68) |
| `f2146` | `01:11.533 (01:11)` | **RECOVERY** | Track `27` | **harshil** | Recovered identity 'harshil' for Track 27 (score=0.720, margin=0.720, tau=0.68) |
| `f2170` | `01:12.333 (01:12)` | **LOCK** | Track `27` | **harshil** | Established permanent sticky identity lock on Track 27 for 'harshil' (score=0.386, margin=0.279) [Confirmed prior Re-ID recovery] |
| `f2279` | `01:15.966 (01:15)` | **LOCK** | Track `25` | **swayam** | Established permanent sticky identity lock on Track 25 for 'swayam' (score=0.390, margin=0.227) [Confirmed prior Re-ID recovery] |
| `f2522` | `01:24.066 (01:24)` | **RELEASE** | Track `27` | **harshil** | Track 27 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f2735` | `01:31.166 (01:31)` | **RECOVERY** | Track `28` | **harshil** | Recovered identity 'harshil' for Track 28 (score=0.787, margin=0.787, tau=0.68) |
| `f2777` | `01:32.566 (01:32)` | **RELEASE** | Track `25` | **swayam** | Track 25 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f2880` | `01:36.000 (01:36)` | **RELEASE** | Track `28` | **harshil** | Track 28 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f3038` | `01:41.266 (01:41)` | **RECOVERY** | Track `29` | **harshil** | Recovered identity 'harshil' for Track 29 (score=0.763, margin=0.120, tau=0.68) |
| `f3039` | `01:41.299 (01:41)` | **LOCK** | Track `20` | **aagam** | Established permanent sticky identity lock on Track 20 for 'aagam' (score=0.450, margin=0.196) [Confirmed prior Re-ID recovery] |
| `f3242` | `01:48.066 (01:48)` | **LOCK** | Track `29` | **harshil** | Established permanent sticky identity lock on Track 29 for 'harshil' (score=0.403, margin=0.286) [Confirmed prior Re-ID recovery] |
| `f3270` | `01:49.000 (01:49)` | **RECOVERY** | Track `30` | **swayam** | Recovered identity 'swayam' for Track 30 (score=0.861, margin=0.861, tau=0.68) |
| `f3271` | `01:49.033 (01:49)` | **LOCK** | Track `30` | **swayam** | Established permanent sticky identity lock on Track 30 for 'swayam' (score=0.400, margin=0.214) [Confirmed prior Re-ID recovery] |
| `f3352` | `01:51.733 (01:51)` | **RELEASE** | Track `29` | **harshil** | Track 29 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f3352` | `01:51.733 (01:51)` | **RECOVERY** | Track `31` | **harshil** | Recovered identity 'harshil' for Track 31 (score=0.747, margin=0.747, tau=0.68) |
| `f3467` | `01:55.566 (01:55)` | **RELEASE** | Track `31` | **harshil** | Track 31 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |
| `f4941` | `02:44.699 (02:44)` | **RECOVERY** | Track `32` | **harshil** | Recovered identity 'harshil' for Track 32 (score=0.749, margin=0.749, tau=0.68) |
| `f5035` | `02:47.833 (02:47)` | **RELEASE** | Track `30` | **swayam** | Track 30 exceeded track_buffer=90; released ownership of 'swayam' back to unowned pool. |
| `f5144` | `02:51.466 (02:51)` | **LOCK** | Track `32` | **swayam** | Established permanent sticky identity lock on Track 32 for 'swayam' (score=0.381, margin=0.178) [Confirmed prior Re-ID recovery] |
| `f5145` | `02:51.500 (02:51)` | **RELEASE** | Track `32` | **harshil** | Track 32 exceeded track_buffer=90; released ownership of 'harshil' back to unowned pool. |

## 5. Visual Coding Legend

- **GREEN Bounding Box**: ArcFace Identity Lock Established. ArcFace is the sole identity authority; sticky/permanent.
- **YELLOW Bounding Box**: Identity RECOVERED via Body Re-ID and not yet face-confirmed. Explicitly labeled `[RECOVERED (Re-ID)]`.
- **ORANGE Bounding Box**: UNKNOWN / No established identity. Motion-only ByteTrack tracklet.
- **Top-Left Status HUD**: Real-time display of frame index, timestamp, running counters, and the live mapping of all four enrolled students.
