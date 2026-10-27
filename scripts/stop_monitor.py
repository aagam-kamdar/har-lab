"""
scripts/stop_monitor.py - Gracefully stop headless monitor.
"""
import json
import os
import time
import psutil

LOCK_FILE = "monitor.lock"

def main():
    if not os.path.exists(LOCK_FILE):
        print("[INFO] Monitor is not running (no lock file found).")
        return

    try:
        with open(LOCK_FILE, "r") as f:
            data = json.load(f)
        pid = data.get("pid")
    except Exception:
        pid = None

    if not pid or not psutil.pid_exists(pid):
        print("[INFO] Monitor is not running (stale lock file). Cleaning up lock.")
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
        return

    proc = psutil.Process(pid)
    print(f"[STOP] Stopping monitor process (PID {pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=4.0)
        print("[OK] Monitor terminated gracefully.")
    except psutil.TimeoutExpired:
        print("[WARN] Monitor did not terminate in time. Forcing kill...")
        proc.kill()
        print("[OK] Monitor killed.")

    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass

if __name__ == "__main__":
    main()
