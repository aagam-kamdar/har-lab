import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, yaml, numpy as np
from ultralytics import YOLO
from collections import defaultdict
from lab_monitor import PersonTracker, extract_person_features, raw_label_person, DEFAULT_TH

th = DEFAULT_TH.copy()
with open('thresholds.yaml', 'r') as f:
    th.update(yaml.safe_load(f))

pose_model = YOLO('yolo26n-pose.pt' if os.path.exists('yolo26n-pose.pt') else 'yolov8n-pose.pt')

def run_eval(th_eval, max_f=300, stride=15):
    cap = cv2.VideoCapture('talking.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f_idx = 0
    persons = {}
    preds = []
    while cap.isOpened() and f_idx < max_f:
        ret, frame = cap.read()
        if not ret: break
        if f_idx % stride == 0:
            now = f_idx / fps
            res_pose = pose_model(frame, verbose=False)[0]
            if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
                kpts = res_pose.keypoints.xy.cpu().numpy()
                confs = res_pose.keypoints.conf.cpu().numpy()
                all_f = {}
                for p_i in range(len(kpts)):
                    pid = f'p_{p_i}'
                    if pid not in persons: persons[pid] = PersonTracker(pid, pid, 10)
                    feat = extract_person_features(persons[pid], kpts[p_i], confs[p_i], {}, now, th_eval)
                    if feat: all_f[pid] = feat
                for pid, feat in all_f.items():
                    lbl, rsn = raw_label_person(persons[pid], feat, all_f, persons, now, th_eval)
                    preds.append(lbl)
        f_idx += 1
    cap.release()
    c = defaultdict(int)
    for p in preds: c[p] += 1
    return len(preds), c

th1 = th.copy()
th1['discuss_min_turn_s'] = 2.0
n1, c1 = run_eval(th1)

th2 = th.copy()
th2['discuss_min_turn_s'] = 0.0
n2, c2 = run_eval(th2)

pct1 = c1['discussing'] / n1 * 100.0 if n1 > 0 else 0.0
pct2 = c2['discussing'] / n2 * 100.0 if n2 > 0 else 0.0

print(f'Total samples: {n1}')
print(f'With 2.0s hold : discussing={c1["discussing"]}/{n1} ({pct1:.1f}%), idle={c1["idle"]}, active_work={c1["active_work"]}')
print(f'With 0.0s hold : discussing={c2["discussing"]}/{n2} ({pct2:.1f}%), idle={c2["idle"]}, active_work={c2["active_work"]}')
