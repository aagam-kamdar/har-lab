import urllib.request
import json
import time
import subprocess
import sys

def main():
    print("Launching app_api.py on port 5002...")
    proc = subprocess.Popen([sys.executable, "app_api.py"])
    time.sleep(2.5)

    try:
        # 1. Test GET /
        with urllib.request.urlopen("http://127.0.0.1:5002/") as resp:
            content = resp.read()
            print(f"GET / -> Status: {resp.status} | Length: {len(content)}")
            assert b"Lab Activity Intelligence Monitor" in content
        
        # 2. Test GET /api/status
        with urllib.request.urlopen("http://127.0.0.1:5002/api/status") as resp:
            data = json.loads(resp.read().decode())
            print(f"GET /api/status -> monitor_online: {data.get('monitor_online')} | seats count: {len(data.get('seats', []))}")
            for s in data.get("seats", []):
                print(f"   Seat {s['seat_id']} ({s['person_id']}): {s['state']} | {s['productivity_pct']}% | {s['reason']}")
                
        # 3. Test GET /api/summary
        with urllib.request.urlopen("http://127.0.0.1:5002/api/summary") as resp:
            summary = json.loads(resp.read().decode())
            print(f"GET /api/summary -> Class prod: {summary.get('class_productivity_pct')}%")

        # 4. Test GET /api/config
        with urllib.request.urlopen("http://127.0.0.1:5002/api/config") as resp:
            cfg = json.loads(resp.read().decode())
            print(f"GET /api/config -> Configured seats: {list(cfg.get('seats', {}).keys())}")

        print("\n[SUCCESS] ALL LIVE HTTP ENDPOINT TESTS PASSED SUCCESSFULLY!")
    finally:
        print("Terminating server...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
