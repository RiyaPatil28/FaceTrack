#!/usr/bin/env python3
"""
Fix database schema for face recognition
"""

import sqlite3
from datetime import datetime

def fix_database():
    """Fix database schema to support face recognition"""
    conn = sqlite3.connect('attendance_system.db')
    cursor = conn.cursor()
    
    # Check if face_encoding column exists
    cursor.execute("PRAGMA table_info(employees)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'face_encoding' not in columns:
        print("Adding face_encoding column...")
        cursor.execute("ALTER TABLE employees ADD COLUMN face_encoding BLOB")
        conn.commit()
        print("✓ Added face_encoding column")
    else:
        print("✓ face_encoding column already exists")
    
    # Check if face_model column exists (legacy)
    if 'face_model' not in columns:
        cursor.execute("ALTER TABLE employees ADD COLUMN face_model BLOB")
        conn.commit()
        print("✓ Added face_model column for compatibility")
    
    conn.close()
    print("Database schema updated successfully!")

if __name__ == "__main__":
    fix_database()