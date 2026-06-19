import sqlite3
import os

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.db')

print("=" * 60)
print("  Fake News Detection System - Data Viewer")
print("=" * 60)

# -- Check Local SQLite DB ----------------------------------
print("\n[LOCAL] Checking local SQLite database (predictions.db)...")

if not os.path.exists(db_path):
    print("[WARN] predictions.db not found locally.")
    print("       Run the Flask app first and make at least one prediction.")
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
            print("  (No predictions made yet)")
    except Exception as e:
        print(f"[ERR] SQLite error: {e}")

print("\n" + "=" * 60)
print("  Summary:")
print("  - All prediction history is stored in predictions.db (SQLite)")
print("  - History is available locally and on Vercel (in /tmp)")
print("=" * 60)