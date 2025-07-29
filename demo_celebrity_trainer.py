#!/usr/bin/env python3
"""
Demo Celebrity Trainer - Creates demo employees from dataset celebrity names
Since the actual image files are not available, this creates demo entries
"""

import pandas as pd
import sqlite3
from datetime import datetime
import numpy as np

def create_demo_celebrities():
    """Create demo celebrity employees from the dataset names"""
    
    # Read the dataset to get celebrity names
    csv_path = "attached_assets/Dataset_1753783807479.csv"
    df = pd.read_csv(csv_path)
    
    # Get unique celebrities
    celebrities = df['label'].unique()
    print(f"Found {len(celebrities)} unique celebrities from dataset")
    
    # Clear existing data
    conn = sqlite3.connect('attendance_system.db')
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM employees WHERE employee_id LIKE 'CELEB%'")
    cursor.execute("DELETE FROM attendance WHERE employee_id LIKE 'CELEB%'")
    print("✓ Cleared existing celebrity data")
    
    # Add celebrities as demo employees (without face training for now)
    added_count = 0
    
    for i, celebrity in enumerate(celebrities[:20], 1):  # Limit to 20 for demo
        employee_id = f"CELEB{i:03d}"
        
        try:
            cursor.execute('''
                INSERT INTO employees 
                (employee_id, name, department, created_date)
                VALUES (?, ?, ?, ?)
            ''', (employee_id, celebrity, 'Entertainment', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            print(f"✓ Added {celebrity} ({employee_id})")
            added_count += 1
            
        except Exception as e:
            print(f"✗ Failed to add {celebrity}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n🎉 Successfully added {added_count} celebrity employees!")
    print("Note: Face recognition training requires the actual image files.")
    print("The employees are added to the database and will appear in the employee list.")

if __name__ == "__main__":
    create_demo_celebrities()