#!/usr/bin/env python3
"""
Mobile API Integration for Smart Attendance System
Provides REST API endpoints for mobile app connectivity
"""

from flask import Flask, jsonify, request, send_file
import sqlite3
import json
from datetime import datetime, date, timedelta
import pandas as pd
import io
import base64
from PIL import Image
import numpy as np
import cv2
import os

app = Flask(__name__)

class MobileAttendanceAPI:
    def __init__(self, db_path='attendance_system.db'):
        self.db_path = db_path
    
    def get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    def get_all_employees(self):
        """Get all employees for mobile display"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT employee_id, name, department, created_date
                FROM employees 
                ORDER BY name
            ''')
            
            employees = []
            for row in cursor.fetchall():
                employees.append({
                    'id': row['employee_id'],
                    'name': row['name'],
                    'department': row['department'],
                    'created_date': row['created_date']
                })
            
            conn.close()
            return employees
            
        except Exception as e:
            print(f"Error getting employees: {e}")
            return []
    
    def get_attendance_by_date_range(self, start_date, end_date):
        """Get attendance records for date range"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT a.id, a.employee_id, a.check_in_time, a.date, 
                       e.name, e.department, COALESCE(a.confidence, 0.0) as confidence
                FROM attendance a
                JOIN employees e ON a.employee_id = e.employee_id
                WHERE a.date BETWEEN ? AND ?
                ORDER BY a.date DESC, a.check_in_time DESC
            ''', (start_date, end_date))
            
            attendance = []
            for row in cursor.fetchall():
                attendance.append({
                    'id': row['id'],
                    'employee_id': row['employee_id'],
                    'employee_name': row['name'],
                    'department': row['department'],
                    'check_in_time': row['check_in_time'],
                    'date': row['date'],
                    'confidence': round(row['confidence'], 1)
                })
            
            conn.close()
            return attendance
            
        except Exception as e:
            print(f"Error getting attendance: {e}")
            return []
    
    def get_today_stats(self):
        """Get today's attendance statistics"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            today = date.today().strftime('%Y-%m-%d')
            
            # Get total employees
            cursor.execute('SELECT COUNT(*) as total FROM employees')
            total_employees = cursor.fetchone()['total']
            
            # Get today's attendance
            cursor.execute('''
                SELECT COUNT(*) as present 
                FROM attendance 
                WHERE date = ?
            ''', (today,))
            present_today = cursor.fetchone()['present']
            
            # Get attendance rate
            attendance_rate = (present_today / total_employees * 100) if total_employees > 0 else 0
            
            # Get recent attendance (last 5)
            cursor.execute('''
                SELECT a.employee_id, e.name, a.check_in_time, a.confidence
                FROM attendance a
                JOIN employees e ON a.employee_id = e.employee_id
                WHERE a.date = ?
                ORDER BY a.check_in_time DESC
                LIMIT 5
            ''', (today,))
            
            recent_attendance = []
            for row in cursor.fetchall():
                recent_attendance.append({
                    'employee_id': row['employee_id'],
                    'name': row['name'],
                    'check_in_time': row['check_in_time'],
                    'confidence': round(row['confidence'], 1)
                })
            
            conn.close()
            
            return {
                'total_employees': total_employees,
                'present_today': present_today,
                'attendance_rate': round(attendance_rate, 1),
                'recent_attendance': recent_attendance,
                'date': today
            }
            
        except Exception as e:
            print(f"Error getting today's stats: {e}")
            return {}
    
    def get_employee_attendance_history(self, employee_id, days=30):
        """Get attendance history for specific employee"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            start_date = (date.today() - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = date.today().strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT a.date, a.check_in_time, a.confidence, e.name, e.department
                FROM attendance a
                JOIN employees e ON a.employee_id = e.employee_id
                WHERE a.employee_id = ? AND a.date BETWEEN ? AND ?
                ORDER BY a.date DESC
            ''', (employee_id, start_date, end_date))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'date': row['date'],
                    'check_in_time': row['check_in_time'],
                    'confidence': round(row['confidence'], 1),
                    'employee_name': row['name'],
                    'department': row['department']
                })
            
            conn.close()
            return history
            
        except Exception as e:
            print(f"Error getting employee history: {e}")
            return []

# Initialize API instance
api = MobileAttendanceAPI()

# API Endpoints
@app.route('/api/status', methods=['GET'])
def api_status():
    """API health check"""
    return jsonify({
        'status': 'active',
        'message': 'Smart Attendance Mobile API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Get all employees"""
    employees = api.get_all_employees()
    return jsonify({
        'success': True,
        'data': employees,
        'count': len(employees)
    })

@app.route('/api/attendance/today', methods=['GET'])
def get_today_attendance():
    """Get today's attendance statistics"""
    stats = api.get_today_stats()
    return jsonify({
        'success': True,
        'data': stats
    })

@app.route('/api/attendance/range', methods=['GET'])
def get_attendance_range():
    """Get attendance for date range"""
    start_date = request.args.get('start_date', date.today().strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', date.today().strftime('%Y-%m-%d'))
    
    attendance = api.get_attendance_by_date_range(start_date, end_date)
    return jsonify({
        'success': True,
        'data': attendance,
        'count': len(attendance),
        'date_range': {
            'start': start_date,
            'end': end_date
        }
    })

@app.route('/api/employee/<employee_id>/history', methods=['GET'])
def get_employee_history(employee_id):
    """Get attendance history for specific employee"""
    days = int(request.args.get('days', 30))
    history = api.get_employee_attendance_history(employee_id, days)
    
    return jsonify({
        'success': True,
        'data': history,
        'employee_id': employee_id,
        'days': days,
        'count': len(history)
    })

@app.route('/api/attendance/export', methods=['GET'])
def export_attendance():
    """Export attendance data as CSV"""
    start_date = request.args.get('start_date', date.today().strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', date.today().strftime('%Y-%m-%d'))
    
    attendance = api.get_attendance_by_date_range(start_date, end_date)
    
    if not attendance:
        return jsonify({'success': False, 'message': 'No data found'})
    
    # Convert to DataFrame for CSV export
    df = pd.DataFrame(attendance)
    
    # Create CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    # Create file-like object for download
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    
    filename = f"attendance_{start_date}_to_{end_date}.csv"
    
    return send_file(
        mem,
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv'
    )

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get comprehensive dashboard statistics"""
    try:
        conn = api.get_db_connection()
        cursor = conn.cursor()
        
        # Today's stats
        today = date.today().strftime('%Y-%m-%d')
        cursor.execute('SELECT COUNT(*) as total FROM employees')
        total_employees = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(*) as present FROM attendance WHERE date = ?', (today,))
        present_today = cursor.fetchone()['present']
        
        # This week's stats
        week_start = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')
        cursor.execute('''
            SELECT COUNT(DISTINCT employee_id) as unique_present 
            FROM attendance 
            WHERE date >= ?
        ''', (week_start,))
        present_this_week = cursor.fetchone()['unique_present']
        
        # This month's stats
        month_start = date.today().replace(day=1).strftime('%Y-%m-%d')
        cursor.execute('''
            SELECT COUNT(DISTINCT employee_id) as unique_present 
            FROM attendance 
            WHERE date >= ?
        ''', (month_start,))
        present_this_month = cursor.fetchone()['unique_present']
        
        # Department-wise attendance today
        cursor.execute('''
            SELECT e.department, COUNT(*) as count
            FROM attendance a
            JOIN employees e ON a.employee_id = e.employee_id
            WHERE a.date = ?
            GROUP BY e.department
            ORDER BY count DESC
        ''', (today,))
        
        dept_stats = []
        for row in cursor.fetchall():
            dept_stats.append({
                'department': row['department'],
                'present': row['count']
            })
        
        # Recent attendance (last 10)
        cursor.execute('''
            SELECT a.employee_id, e.name, e.department, a.check_in_time, a.confidence
            FROM attendance a
            JOIN employees e ON a.employee_id = e.employee_id
            WHERE a.date = ?
            ORDER BY a.check_in_time DESC
            LIMIT 10
        ''', (today,))
        
        recent = []
        for row in cursor.fetchall():
            recent.append({
                'employee_id': row['employee_id'],
                'name': row['name'],
                'department': row['department'],
                'check_in_time': row['check_in_time'],
                'confidence': round(row['confidence'], 1)
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'today': {
                    'total_employees': total_employees,
                    'present': present_today,
                    'attendance_rate': round((present_today / total_employees * 100) if total_employees > 0 else 0, 1)
                },
                'this_week': {
                    'unique_present': present_this_week,
                    'attendance_rate': round((present_this_week / total_employees * 100) if total_employees > 0 else 0, 1)
                },
                'this_month': {
                    'unique_present': present_this_month,
                    'attendance_rate': round((present_this_month / total_employees * 100) if total_employees > 0 else 0, 1)
                },
                'department_stats': dept_stats,
                'recent_attendance': recent,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)