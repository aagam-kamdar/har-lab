"""
scripts/start_monitor.py - Start headless monitor with single-instance verification.
"""
import json
import os
import subprocess
import sys
import time
import psutil

LOCK_FILE = "monitor.lock"

def main():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                data = json.load(f)
            pid = data.get("pid")
            if pid and psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                if proc.is_running() and "python" in proc.name().lower():
                    print(f"[WARN] Monitor is already running with PID {pid}.")
                    return
        except Exception:
            pass

    cmd = [sys.executable, "lab_monitor.py"] + sys.argv[1:]
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    
    print(f"Starting monitor: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, creationflags=creationflags)
    time.sleep(1.0)
    
    if proc.poll() is None:
        print(f"[OK] Monitor successfully started with PID {proc.pid}")
    else:
        print(f"[ERROR] Monitor failed to start (exit code {proc.returncode})")

if __name__ == "__main__":
    main()
