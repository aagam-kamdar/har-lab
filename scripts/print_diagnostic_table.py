import json

d = json.load(open("test_outputs/harshil_f52_vs_f960_diagnostic.json"))
orig = {x["frame"]: x for x in d["original_pipeline"]}
adapt = {x["frame"]: x for x in d["adapter_pipeline"]}

print(f"{'Frame':<5} | {'Orig Det':<12} | {'Orig Box':<22} | {'Orig Face':<16} | {'Orig State':<18} | {'Adapt Det':<10} | {'Adapt Box':<22} | {'Adapt Display':<25}")
print("-" * 145)
for f in range(0, 101, 5):
    o = orig.get(f, {})
    a = adapt.get(f, {})
    o_det = "yolo26n-pose" if o else "N/A"
    o_box = str(o.get("bbox", "N/A"))
    o_face = f"{o.get('face_top_match')}:{o.get('face_score')}" if o.get("face_detected") else "No Face"
    o_lock = f"{o.get('locked_name')} (LOCKED)" if o.get("locked") else (f"{o.get('body_match')} (REC)" if o.get("body_match") else "None")
    a_det = "yolo26n" if a else "N/A"
    a_box = str(a.get("bbox", "N/A"))
    a_disp = str(a.get("display_name", "N/A"))
    print(f"f{f:<4} | {o_det:<12} | {o_box:<22} | {o_face:<16} | {o_lock:<18} | {a_det:<10} | {a_box:<22} | {a_disp:<25}")
