import os
import sys
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATABASE_PATH

def update_st_labels():
    st_file = os.path.join(PROJECT_ROOT, "data", "st股代码.txt")
    if not os.path.exists(st_file):
        print("st股代码.txt not found.")
        return

    with open(st_file, "r", encoding="utf-8") as f:
        st_codes = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(st_codes)} ST stock codes from file.")

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Check if is_st column exists in stock_info_extended, if not, add it
    cursor.execute("PRAGMA table_info(stock_info_extended)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "is_st" not in columns:
        print("Adding is_st column to stock_info_extended...")
        try:
            cursor.execute("ALTER TABLE stock_info_extended ADD COLUMN is_st INTEGER DEFAULT 0")
        except sqlite3.OperationalError as e:
            print(f"Error adding column: {e}")

    # Set all is_st to 0 first
    cursor.execute("UPDATE stock_info_extended SET is_st = 0")
    
    # Update is_st to 1 for ST stocks
    if st_codes:
        placeholders = ",".join(["?"] * len(st_codes))
        cursor.execute(f"UPDATE stock_info_extended SET is_st = 1 WHERE code IN ({placeholders})", st_codes)
    
    conn.commit()
    print(f"Updated {cursor.rowcount} ST stock labels in database.")
    conn.close()

if __name__ == "__main__":
    update_st_labels()
