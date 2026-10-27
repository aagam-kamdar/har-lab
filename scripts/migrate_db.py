import sqlite3

def migrate(db_path="lab_events.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Check tables
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    tables = dict(cur.fetchall())
    print("Existing tables:", list(tables.keys()))
    
    # Check events table columns
    if "events" in tables:
        cur.execute("PRAGMA table_info(events)")
        cols = [c[1] for c in cur.fetchall()]
        print("Events cols:", cols)
        if "person_id" not in cols:
            print("Adding person_id to events...")
            cur.execute("ALTER TABLE events ADD COLUMN person_id TEXT")
            if "student" in cols:
                cur.execute("UPDATE events SET person_id = student WHERE person_id IS NULL")
        if "camera_id" not in cols:
            print("Adding camera_id to events...")
            cur.execute("ALTER TABLE events ADD COLUMN camera_id TEXT DEFAULT 'cam_0'")
    
    # Check alerts table columns
    if "alerts" in tables:
        cur.execute("PRAGMA table_info(alerts)")
        cols = [c[1] for c in cur.fetchall()]
        print("Alerts cols:", cols)
        if "person_id" not in cols:
            print("Adding person_id to alerts...")
            cur.execute("ALTER TABLE alerts ADD COLUMN person_id TEXT")
            if "student" in cols:
                cur.execute("UPDATE alerts SET person_id = student WHERE person_id IS NULL")
                
    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
