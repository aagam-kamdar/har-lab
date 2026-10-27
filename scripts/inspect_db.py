import sqlite3

conn = sqlite3.connect('lab_events.db')
cur = conn.cursor()
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('Existing tables in lab_events.db:', tables)
for t in tables:
    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
    print(f"  {t}: {cols}")
conn.close()
