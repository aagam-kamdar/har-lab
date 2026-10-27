import subprocess
import sys
import time
import sqlite3

def run_test():
    print("Testing person-centric lab_monitor.py replay mode...")
    proc = subprocess.Popen([sys.executable, "lab_monitor.py", "--replay", "--source", "new lab video.mp4", "--force"])
    
    # Run for 20 seconds
    time.sleep(20)
    print("Sending SIGTERM to monitor...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        
    print("Monitor finished. Checking lab_events.db...")
    conn = sqlite3.connect("lab_events.db")
    cur = conn.cursor()
    
    # Person status
    cur.execute("SELECT person_id, display_name, track_id, state, productivity_pct, desk_zone, reason FROM person_status")
    p_rows = cur.fetchall()
    print(f"\n--- PERSON STATUS ({len(p_rows)} persons) ---")
    for r in p_rows:
        print("  ->", r)
        
    # Events
    cur.execute("SELECT count(*) FROM events")
    print("\nEvents count:", cur.fetchone()[0])
    
    # Recent events
    cur.execute("SELECT person_id, label, duration_sec, desk_zone, reason FROM events ORDER BY id DESC LIMIT 5")
    print("Recent events:")
    for ev in cur.fetchall():
        print("  *", ev)
        
    # Heartbeat
    cur.execute("SELECT last_update, fps, active_seats, status FROM monitor_heartbeat")
    print("\nHeartbeat:", cur.fetchone())
    conn.close()

if __name__ == "__main__":
    run_test()
