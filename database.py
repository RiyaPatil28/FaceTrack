import sqlite3
import numpy as np
import pickle
from datetime import datetime, date
import os

class Database:
    def __init__(self, db_path='attendance_system.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create employees table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                face_encoding BLOB NOT NULL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                date DATE DEFAULT (date('now')),
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        # Create index for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_attendance_date 
            ON attendance (date)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_attendance_employee 
            ON attendance (employee_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_employee(self, employee_id, name, department, face_encoding):
        """Add a new employee to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize face encoding
            face_encoding_blob = pickle.dumps(face_encoding)
            
            cursor.execute('''
                INSERT INTO employees (employee_id, name, department, face_encoding)
                VALUES (?, ?, ?, ?)
            ''', (employee_id, name, department, face_encoding_blob))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.IntegrityError:
            # Employee ID already exists
            if 'conn' in locals():
                conn.close()
            return False
        except Exception as e:
            print(f"Error adding employee: {e}")
            if 'conn' in locals():
                conn.close()
            return False
    
    def get_employee(self, employee_id):
        """Get employee information by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT employee_id, name, department, created_date, face_encoding
                FROM employees 
                WHERE employee_id = ?
            ''', (employee_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                # Deserialize face encoding
                face_encoding = pickle.loads(result[4])
                return (result[0], result[1], result[2], result[3], face_encoding)
            return None
            
        except Exception as e:
            print(f"Error getting employee: {e}")
            return None
    
    def get_all_employees(self):
        """Get all employees (without face encodings for display)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT employee_id, name, department, created_date, face_encoding
                FROM employees 
                ORDER BY name
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            # Return without face encodings for general use
            return [(r[0], r[1], r[2], r[3], "encoded") for r in results]
            
        except Exception as e:
            print(f"Error getting all employees: {e}")
            return []
    
    def get_all_employees_with_encodings(self):
        """Get all employees with their face encodings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT employee_id, name, department, created_date, face_encoding
                FROM employees 
                ORDER BY name
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            # Deserialize face encodings
            employees_with_encodings = []
            for result in results:
                face_encoding = pickle.loads(result[4])
                employees_with_encodings.append((
                    result[0], result[1], result[2], result[3], face_encoding
                ))
            
            return employees_with_encodings
            
        except Exception as e:
            print(f"Error getting employees with encodings: {e}")
            return []
    
    def remove_employee(self, employee_id):
        """Remove an employee from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Also remove attendance records for this employee
            cursor.execute('DELETE FROM attendance WHERE employee_id = ?', (employee_id,))
            cursor.execute('DELETE FROM employees WHERE employee_id = ?', (employee_id,))
            
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            return rows_affected > 0
            
        except Exception as e:
            print(f"Error removing employee: {e}")
            return False
    
    def mark_attendance(self, employee_id):
        """Mark attendance for an employee"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if employee exists
            cursor.execute('SELECT employee_id FROM employees WHERE employee_id = ?', (employee_id,))
            if not cursor.fetchone():
                conn.close()
                return False
            
            # Check if already marked today
            today = date.today().strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE employee_id = ? AND date = ?
            ''', (employee_id, today))
            
            if cursor.fetchone():
                # Already marked today
                conn.close()
                return False
            
            # Mark attendance
            cursor.execute('''
                INSERT INTO attendance (employee_id, check_in_time, date)
                VALUES (?, ?, ?)
            ''', (employee_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), today))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False
    
    def get_attendance_by_date(self, date_str):
        """Get attendance records for a specific date"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, employee_id, check_in_time, date
                FROM attendance 
                WHERE date = ?
                ORDER BY check_in_time
            ''', (date_str,))
            
            results = cursor.fetchall()
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error getting attendance by date: {e}")
            return []
    
    def get_attendance_range(self, start_date, end_date):
        """Get attendance records for a date range"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, employee_id, check_in_time, date
                FROM attendance 
                WHERE date BETWEEN ? AND ?
                ORDER BY date, check_in_time
            ''', (start_date, end_date))
            
            results = cursor.fetchall()
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error getting attendance range: {e}")
            return []
    
    def get_all_attendance(self):
        """Get all attendance records"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, employee_id, check_in_time, date
                FROM attendance 
                ORDER BY date DESC, check_in_time DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error getting all attendance: {e}")
            return []
    
    def get_employee_attendance_count(self, employee_id, start_date=None, end_date=None):
        """Get attendance count for a specific employee"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if start_date and end_date:
                cursor.execute('''
                    SELECT COUNT(*) FROM attendance 
                    WHERE employee_id = ? AND date BETWEEN ? AND ?
                ''', (employee_id, start_date, end_date))
            else:
                cursor.execute('''
                    SELECT COUNT(*) FROM attendance 
                    WHERE employee_id = ?
                ''', (employee_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 0
            
        except Exception as e:
            print(f"Error getting employee attendance count: {e}")
            return 0
    
    def is_attendance_marked_today(self, employee_id):
        """Check if attendance is already marked for today"""
        try:
            today = date.today().strftime('%Y-%m-%d')
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE employee_id = ? AND date = ?
            ''', (employee_id, today))
            
            result = cursor.fetchone()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            print(f"Error checking today's attendance: {e}")
            return False
