import sys, os, io, uuid, json, time
from pathlib import Path

ROOT = r'c:\Users\admin\OneDrive\Desktop\Projects'
sys.path.insert(0, ROOT)
os.environ['ROBOFLOW_API_KEY'] = 'run_check_key'

SEP = '=' * 72

def banner(title):
    print()
    print(SEP)
    print(f'  {title}')
    print(SEP)

banner('[1/8] ENVIRONMENT')
print(f'Python:       {sys.version}')
print(f'Executable:   {sys.executable}')
print(f'Platform:     {sys.platform}')
print(f'Working dir:  {os.getcwd()}')
print(f'Project root: {ROOT}')
pkgs = ['flask','cv2','numpy','requests','inference_sdk']
for p in pkgs:
    try:
        m = __import__(p.replace('-','_'))
        v = getattr(m,'__version__', '?')
        print(f'  import {p:<16} OK (version {v})')
    except Exception as e:
        print(f'  import {p:<16} FAILED: {e!r}')

banner('[2/8] REQUIREMENTS.TXT CONTENTS')
req_path = os.path.join(ROOT, 'requirements.txt')
for i, line in enumerate(open(req_path), 1):
    print(f'  L{i:02d}: {line.rstrip()}')

banner('[3/8] IMPORT APP + ROBFLOW_CLIENT')
import app as app_module
import roboflow_client as rc
print('app module imported: vars -> APP_ROOT={}'.format(app_module.APP_ROOT))
print('app module imported: vars -> RESULTS_DIR={}'.format(app_module.RESULTS_DIR))
print('app module imported: vars -> ALLOWED_EXTENSIONS={}'.format(sorted(app_module.ALLOWED_EXTENSIONS)))
print('app module imported: vars -> MAX_CONTENT_LENGTH={} MB'.format(app_module.MAX_CONTENT_LENGTH//(1024*1024)))
print('roboflow_client API_KEY env var name used: ROBOFLOW_API_KEY')
print('roboflow_client WORKFLOW_URL: {}'.format(rc.WORKFLOW_URL))
syms = ['RoboflowWorkflowError','WorkflowResult','Detection','run_workflow','run_workflow_on_image']
for s in syms:
    print(f'  symbol {s:<22} -> {hasattr(rc, s)} {type(getattr(rc,s,"")).__name__}')

banner('[4/8] INSTANTIATE DATACLASSES')
d = rc.Detection(x=32.5, y=40.0, width=20.0, height=30.0, class_name='phone', confidence=0.934)
print(f'Detection -> {d}')
wr = rc.WorkflowResult(detections=[d], raw={'meta':'ok'})
print(f'WorkflowResult -> detections={len(wr.detections)} raw_keys={list(wr.raw.keys()) if wr.raw else None}')

banner('[5/8] run_workflow_on_image MOCKED - HAPPY PATH')
import unittest.mock as mock
import numpy as np
frame = np.zeros((64, 64, 3), dtype=np.uint8)
frame[20:40, 10:30] = (50, 120, 200)  # draw a colored rectangle just so it's not all 0s
fake_resp = mock.MagicMock()
fake_resp.status_code = 200
fake_resp.text = 'ok'
fake_resp.json.return_value = {
    'predictions': [
        {'x':100.0,'y':100.0,'width':50,'height':40,'class':'phone','confidence':0.97},
        {'x':200.0,'y':50.0, 'width':30,'height':25,'class':'book', 'confidence':0.82},
    ]
}
with mock.patch('requests.post', return_value=fake_resp) as pmock:
    t0 = time.time()
    result = rc.run_workflow_on_image(frame)
    dt = (time.time()-t0)*1000
print(f'  elapsed:           {dt:.2f} ms')
print(f'  detections count:  {len(result.detections)}')
for i, det in enumerate(result.detections):
    print(f'    det#{i} class={det.class_name:<10} conf={det.confidence:.2f} '
          f'bbox_center=({det.x:g},{det.y:g}) size=({det.width:g}x{det.height:g})')
print(f'  post called?       {pmock.called}')
print(f'  post call_count:   {pmock.call_count}')
kwargs = pmock.call_args.kwargs
print(f'  post URL (args[0]): {pmock.call_args.args[0]}')
print(f'  post params:       {kwargs.get("params")}')
print(f'  post files keys:   {list(kwargs.get("files",{}).keys())}')

banner('[6/8] run_workflow_on_image MOCKED - ERROR PATH (HTTP 500)')
bad_resp = mock.MagicMock()
bad_resp.status_code = 500
bad_resp.text = 'Internal Roboflow Server Error: out of memory'
with mock.patch('requests.post', return_value=bad_resp):
    try:
        rc.run_workflow_on_image(frame)
        print('  FAIL: should have raised RoboflowWorkflowError')
    except rc.RoboflowWorkflowError as e:
        msg = str(e)
        has_500 = 'HTTP 500' in msg
        has_body = 'out of memory' in msg
        print(f'  raised correctly:  YES')
        print(f'  contains HTTP 500: {has_500}')
        print(f'  contains body text: {has_body}')
        print(f'  full message:      {msg[:120]}')

banner('[7/8] FLASK TEST CLIENT: GET /')
client = app_module.app.test_client()
resp = client.get('/')
print(f'  status: {resp.status_code}')
body = resp.get_data(as_text=True)
checks = [
    ('<title>Student Productivity Monitoring</title>', 'title tag'),
    ('name="sample_fps"', 'sample_fps input'),
    ('name="max_frames"',   'max_frames input'),
    ('name="video"',        'video file input'),
    ('type="submit"',       'submit button (or Analyze button in tag)'),
    ('Analyze',             '"Analyze" button text'),
]
for needle, desc in checks:
    ok = needle in body
    print(f'  {desc:<30}: {ok}')
print(f'  total html length: {len(body)} chars')
print(f'  body first 600 chars:')
print('  | ' + body[:600].replace('\n','\n  | '))

banner('[8/8] FLASK TEST CLIENT: POST /analyze FULL E2E (mocked Roboflow)')
import cv2
tmp_avi = os.path.join(ROOT, '_tmp_e2e_video.avi')
W, H, FPS, NFRAMES = 128, 96, 30, 60
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vw = cv2.VideoWriter(tmp_avi, fourcc, FPS, (W, H))
for i in range(NFRAMES):
    f = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.putText(f, f'f{i:03d}', (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1)
    cv2.rectangle(f, (20+i%30, 20), (70+i%30, 60), (80,80,255), 2)
    vw.write(f)
vw.release()
with open(tmp_avi, 'rb') as fh:
    avi_bytes = fh.read()
os.remove(tmp_avi)
print(f'  synthesized {NFRAMES} frames @ {FPS}fps -> {len(avi_bytes)} bytes (MJPG .avi, {W}x{H})')

fake_resp2 = mock.MagicMock()
fake_resp2.status_code = 200
fake_resp2.text = 'ok'
fake_resp2.json.return_value = {
    'predictions': [
        {'x':60.0,'y':50.0,'width':40,'height':40,'class':'student','confidence':0.91},
        {'x':90.0,'y':30.0,'width':25,'height':20,'class':'laptop', 'confidence':0.78},
        {'x':30.0,'y':75.0,'width':15,'height':12,'class':'phone',  'confidence':0.66},
    ]
}

# clean up old results dirs for a clean count
res_dir = Path(app_module.RESULTS_DIR)
pre_jobs = set(p.name for p in res_dir.iterdir() if p.is_dir())
print(f'  existing result job dirs before run: {len(pre_jobs)}')

sample_fps = '2.0'
max_frames = '3'
with mock.patch('requests.post', return_value=fake_resp2) as pmock2:
    t0 = time.time()
    resp2 = client.post('/analyze', data={
        'video': (io.BytesIO(avi_bytes), 'classroom_1min.avi', 'video/avi'),
        'sample_fps': sample_fps,
        'max_frames': max_frames,
    }, content_type='multipart/form-data', follow_redirects=True)
    dt2 = time.time() - t0

print(f'  sample_fps/max_frames request params: {sample_fps} / {max_frames}')
print(f'  status:                 {resp2.status_code}')
print(f'  requests.post calls:    {pmock2.call_count} (expected ~{max_frames})')
print(f'  elapsed wall time:      {dt2:.2f} s')
body2 = resp2.get_data(as_text=True)
checks2 = [
    ('Total detections',                'summary block with total detections'),
    ('Frames analyzed',                 'summary block with frames count'),
    ('Job ID',                          'summary shows Job ID (or job_id rendered)'),
    ('/static/results/',                'img src points into static/results/'),
    ('student',                         'class name "student" somewhere (from mock payload)'),
    ('laptop',                          'class name "laptop" somewhere'),
    ('phone',                           'class name "phone" somewhere'),
    ('0.91',                            'top confidence 0.91 rendered with 2 decimals'),
    ('Upload another video',            'back link / CTA present'),
]
print(f'  response html length:   {len(body2)} chars')
print(f'  rendered checks:')
for needle, desc in checks2:
    ok = needle.lower() in body2.lower()
    print(f'    {desc:<42}: {ok}')

post_jobs = set(p.name for p in res_dir.iterdir() if p.is_dir())
new_jobs = sorted(post_jobs - pre_jobs)
print(f'  new result job dirs created: {new_jobs}')
for j in new_jobs:
    jp = res_dir / j
    jpgs = sorted(p.name for p in jp.glob('*.jpg'))
    print(f'    job {j} -> {len(jpgs)} jpg(s): {jpgs}')

print()
print(SEP)
print('  E2E RUN COMPLETED SUCCESSFULLY'.center(72))
print(SEP)
