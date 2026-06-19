import urllib.request
import urllib.error
import json
import sqlite3
import os

KVDB_URL = "https://kvdb.io/4UwJyhr3ootyftcgQJHkMP/history"

print("=" * 60)
print("  Fake News Detection System - Data Viewer")
print("=" * 60)

# -- 1. Check KVDB Cloud ---------------------------------------
print("\n[CLOUD] Checking KVDB Cloud data...")
try:
    req = urllib.request.Request(KVDB_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=8) as r:
        raw = r.read().decode('utf-8')
        if raw.strip():
            data = json.loads(raw)
            print(f"[OK] Found {len(data)} record(s) in KVDB Cloud:\n")
            for i, item in enumerate(data, 1):
                print(f"  [{i}] {item.get('timestamp','N/A')} | "
                      f"{item.get('prediction','?'):10s} | "
                      f"Conf: {item.get('confidence', 0):.2%} | "
                      f"Text: {item.get('text','')[:60]}...")
        else:
            print("[WARN] KVDB key exists but is empty.")
except urllib.error.HTTPError as e:
    if e.code == 404:
        print("[WARN] No data in KVDB yet (404 - key has never been written).")
        print("       Data is saved to KVDB only when the app runs on Vercel.")
    else:
        print(f"[ERR] KVDB HTTP Error {e.code}: {e.reason}")
except Exception as e:
    print(f"[ERR] Could not reach KVDB: {e}")

# -- 2. Check Local SQLite DB ----------------------------------
print("\n[LOCAL] Checking local SQLite database (predictions.db)...")
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.db')

if not os.path.exists(db_path):
    print("[WARN] predictions.db not found locally.")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM history")
        count = cursor.fetchone()[0]
        print(f"[OK] Found {count} record(s) in local SQLite:\n")
        cursor.execute(
            "SELECT input_text, prediction, confidence, timestamp FROM history ORDER BY id DESC LIMIT 20"
        )
        rows = cursor.fetchall()
        for i, row in enumerate(rows, 1):
            print(f"  [{i}] {row[3]} | {row[1]:10s} | Conf: {row[2]:.2%} | Text: {row[0][:60]}...")
        conn.close()
        if count == 0:
            print("  (No predictions made yet locally either)")
    except Exception as e:
        print(f"[ERR] SQLite error: {e}")

print("\n" + "=" * 60)
print("  Summary:")
print("  - KVDB is used when app runs on Vercel (production)")
print("  - SQLite is used when app runs locally")
print("=" * 60)