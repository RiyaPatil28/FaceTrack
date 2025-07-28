import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, date
import time
from PIL import Image
import io
import base64
import sqlite3
import pickle

# Simple database operations without complex dependencies
class SimpleDatabase:
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
        
        conn.commit()
        conn.close()
    
    def add_employee(self, employee_id, name, department):
        """Add a new employee to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO employees (employee_id, name, department)
                VALUES (?, ?, ?)
            ''', (employee_id, name, department))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.IntegrityError:
            if 'conn' in locals():
                conn.close()
            return False
        except Exception as e:
            print(f"Error adding employee: {e}")
            if 'conn' in locals():
                conn.close()
            return False
    
    def get_all_employees(self):
        """Get all employees"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT employee_id, name, department, created_date
                FROM employees 
                ORDER BY name
            ''')
            
            results = cursor.fetchall()
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error getting all employees: {e}")
            return []
    
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
                SELECT a.id, a.employee_id, a.check_in_time, a.date, e.name
                FROM attendance a
                JOIN employees e ON a.employee_id = e.employee_id
                WHERE a.date = ?
                ORDER BY a.check_in_time
            ''', (date_str,))
            
            results = cursor.fetchall()
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error getting attendance by date: {e}")
            return []

def main():
    st.set_page_config(
        page_title="Smart Attendance System",
        page_icon="👥",
        layout="wide"
    )
    
    # Initialize database
    db = SimpleDatabase()
    
    st.title("🎯 Smart Attendance System")
    st.markdown("AI-powered face recognition for automated attendance tracking")
    
    # Add demo note
    st.warning("📝 **Demo Mode**: This is a simplified version. Face recognition functionality will be added once dependencies are properly installed.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📸 Live Attendance", "👤 Employee Management", "📊 Reports & Analytics", "⚙️ System Status"]
    )
    
    if page == "📸 Live Attendance":
        live_attendance_page(db)
    elif page == "👤 Employee Management":
        employee_management_page(db)
    elif page == "📊 Reports & Analytics":
        reports_page(db)
    elif page == "⚙️ System Status":
        system_status_page(db)

def live_attendance_page(db):
    st.header("📸 Live Attendance Tracking")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed")
        st.info("📷 Camera functionality will be available once OpenCV and face_recognition packages are installed.")
        
        # Manual attendance marking for demo
        st.subheader("Manual Attendance (Demo)")
        employees = db.get_all_employees()
        
        if employees:
            employee_options = {f"{emp[1]} (ID: {emp[0]})": emp[0] for emp in employees}
            
            selected_employee = st.selectbox(
                "Select employee to mark attendance:",
                options=list(employee_options.keys())
            )
            
            if st.button("✅ Mark Attendance", type="primary"):
                employee_id = employee_options[selected_employee]
                success = db.mark_attendance(employee_id)
                
                if success:
                    st.success(f"✅ Attendance marked for {selected_employee}!")
                    st.rerun()
                else:
                    st.warning("⚠️ Attendance already marked today for this employee!")
        else:
            st.info("No employees registered. Please add employees first.")
    
    with col2:
        st.subheader("📋 Today's Attendance")
        
        # Get today's attendance
        today = date.today().strftime('%Y-%m-%d')
        today_attendance = db.get_attendance_by_date(today)
        
        if today_attendance:
            for record in today_attendance:
                st.info(f"👤 {record[4]}\n⏰ {record[2]}")
        else:
            st.info("No attendance records for today yet.")
        
        # Live statistics
        st.subheader("📈 Live Stats")
        total_employees = len(db.get_all_employees())
        present_today = len(today_attendance)
        
        st.metric("Total Employees", total_employees)
        st.metric("Present Today", present_today)
        
        if total_employees > 0:
            attendance_rate = (present_today / total_employees) * 100
            st.metric("Attendance Rate", f"{attendance_rate:.1f}%")

def employee_management_page(db):
    st.header("👤 Employee Management")
    
    tab1, tab2 = st.tabs(["➕ Add Employee", "👥 View Employees"])
    
    with tab1:
        st.subheader("Add New Employee")
        
        with st.form("add_employee_form"):
            employee_id = st.text_input("Employee ID", help="Unique identifier for the employee")
            employee_name = st.text_input("Employee Name", help="Full name of the employee")
            department = st.text_input("Department", help="Employee's department")
            
            st.info("📸 Photo upload for face recognition will be enabled once dependencies are installed.")
            
            submit_button = st.form_submit_button("Add Employee", type="primary")
            
            if submit_button:
                if employee_id and employee_name:
                    success = db.add_employee(employee_id, employee_name, department)
                    
                    if success:
                        st.success(f"✅ Employee {employee_name} added successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add employee. Employee ID might already exist.")
                else:
                    st.error("❌ Please fill in Employee ID and Name.")
    
    with tab2:
        st.subheader("All Employees")
        
        employees = db.get_all_employees()
        
        if employees:
            # Create a DataFrame for better display
            df_data = []
            for emp in employees:
                df_data.append({
                    'ID': emp[0],
                    'Name': emp[1], 
                    'Department': emp[2],
                    'Added Date': emp[3]
                })
            display_df = pd.DataFrame(df_data)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export employees data
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Employees List",
                data=csv,
                file_name=f"employees_list_{date.today()}.csv",
                mime="text/csv"
            )
        else:
            st.info("No employees registered yet. Add some employees to get started!")

def reports_page(db):
    st.header("📊 Reports & Analytics")
    
    tab1, tab2 = st.tabs(["📅 Daily Report", "📈 Analytics"])
    
    with tab1:
        st.subheader("Daily Attendance Report")
        
        # Date selector
        selected_date = st.date_input("Select Date", value=date.today())
        
        # Get attendance for selected date
        attendance_records = db.get_attendance_by_date(selected_date.strftime('%Y-%m-%d'))
        
        if attendance_records:
            # Create DataFrame
            df_records = []
            for record in attendance_records:
                df_records.append({
                    'Employee ID': record[1],
                    'Employee Name': record[4],
                    'Check-in Time': record[2],
                    'Date': record[3]
                })
            
            if df_records:
                df = pd.DataFrame(df_records)
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Present", len(df_records))
                with col2:
                    total_employees = len(db.get_all_employees())
                    st.metric("Total Employees", total_employees)
                with col3:
                    if total_employees > 0:
                        attendance_rate = (len(df_records) / total_employees) * 100
                        st.metric("Attendance Rate", f"{attendance_rate:.1f}%")
        else:
            st.info(f"No attendance records found for {selected_date}")
    
    with tab2:
        st.subheader("Attendance Analytics")
        st.info("📊 Advanced analytics will be available once all dependencies are installed.")

def system_status_page(db):
    st.header("⚙️ System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Dependencies Status")
        
        # Check core packages
        try:
            import streamlit
            st.success("✅ Streamlit - OK")
        except:
            st.error("❌ Streamlit - Not Available")
        
        try:
            import numpy
            st.success("✅ NumPy - OK")
        except:
            st.error("❌ NumPy - Not Available")
        
        try:
            import pandas
            st.success("✅ Pandas - OK")
        except:
            st.error("❌ Pandas - Not Available")
        
        try:
            import cv2
            st.success("✅ OpenCV - OK")
        except:
            st.warning("⚠️ OpenCV - Not Available (required for camera)")
        
        try:
            import face_recognition
            st.success("✅ Face Recognition - OK")
        except:
            st.warning("⚠️ Face Recognition - Not Available (required for AI detection)")
    
    with col2:
        st.subheader("🗄️ Database Status")
        
        try:
            employees = db.get_all_employees()
            st.success("✅ Database Connection - OK")
            st.info(f"📊 Total Employees: {len(employees)}")
            
            today = date.today().strftime('%Y-%m-%d')
            today_attendance = db.get_attendance_by_date(today)
            st.info(f"📅 Today's Attendance: {len(today_attendance)}")
            
        except Exception as e:
            st.error(f"❌ Database Error: {str(e)}")

if __name__ == "__main__":
    main()