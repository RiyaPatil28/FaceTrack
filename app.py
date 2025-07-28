import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date
import time
from PIL import Image
import io
import base64

from database import Database
from face_recognition_utils import FaceRecognitionUtils
from attendance_manager import AttendanceManager

# Initialize database and utilities
@st.cache_resource
def initialize_system():
    db = Database()
    face_utils = FaceRecognitionUtils()
    attendance_mgr = AttendanceManager(db, face_utils)
    return db, face_utils, attendance_mgr

def main():
    st.set_page_config(
        page_title="Smart Attendance System",
        page_icon="👥",
        layout="wide"
    )
    
    # Initialize system components
    db, face_utils, attendance_mgr = initialize_system()
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'last_recognition_time' not in st.session_state:
        st.session_state.last_recognition_time = {}
    
    st.title("🎯 Smart Attendance System")
    st.markdown("AI-powered face recognition for automated attendance tracking")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📸 Live Attendance", "👤 Employee Management", "📊 Reports & Analytics", "⚙️ System Status"]
    )
    
    if page == "📸 Live Attendance":
        live_attendance_page(attendance_mgr, face_utils)
    elif page == "👤 Employee Management":
        employee_management_page(db, face_utils)
    elif page == "📊 Reports & Analytics":
        reports_page(db)
    elif page == "⚙️ System Status":
        system_status_page(db)

def live_attendance_page(attendance_mgr, face_utils):
    st.header("📸 Live Attendance Tracking")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed")
        
        # Camera controls
        camera_col1, camera_col2 = st.columns(2)
        with camera_col1:
            if st.button("🎥 Start Camera", type="primary"):
                st.session_state.camera_active = True
        with camera_col2:
            if st.button("⏹️ Stop Camera"):
                st.session_state.camera_active = False
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        
        if st.session_state.camera_active:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Cannot access camera. Please check camera permissions.")
                    return
                
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Real-time processing loop
                frame_count = 0
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to capture frame from camera")
                        break
                    
                    # Process every 5th frame for performance
                    if frame_count % 5 == 0:
                        # Detect and recognize faces
                        processed_frame, recognized_faces = attendance_mgr.process_frame(frame)
                        
                        # Mark attendance for recognized faces
                        for face_info in recognized_faces:
                            employee_id = face_info['employee_id']
                            confidence = face_info['confidence']
                            
                            # Prevent duplicate entries within 30 seconds
                            current_time = time.time()
                            last_time = st.session_state.last_recognition_time.get(employee_id, 0)
                            
                            if current_time - last_time > 30:  # 30 second cooldown
                                success = attendance_mgr.mark_attendance(employee_id)
                                if success:
                                    st.session_state.last_recognition_time[employee_id] = current_time
                                    # Show success message in sidebar
                                    with col2:
                                        st.success(f"✅ Attendance marked for Employee ID: {employee_id}")
                        
                        # Display processed frame
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    frame_count += 1
                    time.sleep(0.1)  # Small delay for performance
                
                cap.release()
                
            except Exception as e:
                st.error(f"❌ Camera error: {str(e)}")
                st.session_state.camera_active = False
    
    with col2:
        st.subheader("📋 Today's Attendance")
        
        # Get today's attendance
        today_attendance = attendance_mgr.get_today_attendance()
        
        if today_attendance:
            for record in today_attendance:
                employee_name = attendance_mgr.db.get_employee(record[1])
                if employee_name:
                    st.info(f"👤 {employee_name[1]}\n⏰ {record[2]}")
        else:
            st.info("No attendance records for today yet.")
        
        # Live statistics
        st.subheader("📈 Live Stats")
        total_employees = len(attendance_mgr.db.get_all_employees())
        present_today = len(today_attendance)
        
        st.metric("Total Employees", total_employees)
        st.metric("Present Today", present_today)
        
        if total_employees > 0:
            attendance_rate = (present_today / total_employees) * 100
            st.metric("Attendance Rate", f"{attendance_rate:.1f}%")

def employee_management_page(db, face_utils):
    st.header("👤 Employee Management")
    
    tab1, tab2, tab3 = st.tabs(["➕ Add Employee", "👥 View Employees", "🗑️ Remove Employee"])
    
    with tab1:
        st.subheader("Add New Employee")
        
        with st.form("add_employee_form"):
            employee_id = st.text_input("Employee ID", help="Unique identifier for the employee")
            employee_name = st.text_input("Employee Name", help="Full name of the employee")
            department = st.text_input("Department", help="Employee's department")
            
            st.markdown("**Upload Employee Photo**")
            uploaded_file = st.file_uploader(
                "Choose image file", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear front-facing photo of the employee"
            )
            
            submit_button = st.form_submit_button("Add Employee", type="primary")
            
            if submit_button:
                if employee_id and employee_name and uploaded_file:
                    try:
                        # Read and process the uploaded image
                        image = Image.open(uploaded_file)
                        image_array = np.array(image)
                        
                        # Convert to RGB if necessary
                        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        else:
                            image_rgb = image_array
                        
                        # Generate face encoding
                        face_encoding = face_utils.get_face_encoding(image_rgb)
                        
                        if face_encoding is not None:
                            # Save employee to database
                            success = db.add_employee(employee_id, employee_name, department, face_encoding)
                            
                            if success:
                                st.success(f"✅ Employee {employee_name} added successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to add employee. Employee ID might already exist.")
                        else:
                            st.error("❌ No face detected in the uploaded image. Please upload a clear photo with a visible face.")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}")
                else:
                    st.error("❌ Please fill in all fields and upload an image.")
    
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
    
    with tab3:
        st.subheader("Remove Employee")
        
        employees = db.get_all_employees()
        
        if employees:
            employee_options = {f"{emp[1]} (ID: {emp[0]})": emp[0] for emp in employees}
            
            selected_employee = st.selectbox(
                "Select employee to remove:",
                options=list(employee_options.keys())
            )
            
            if st.button("🗑️ Remove Employee", type="secondary"):
                employee_id = employee_options[selected_employee]
                
                # Confirmation
                if st.button("⚠️ Confirm Removal", type="primary"):
                    success = db.remove_employee(employee_id)
                    if success:
                        st.success("✅ Employee removed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to remove employee.")
        else:
            st.info("No employees to remove.")

def reports_page(db):
    st.header("📊 Reports & Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📅 Daily Report", "📈 Analytics", "📥 Export Data"])
    
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
                employee = db.get_employee(record[1])
                if employee:
                    df_records.append({
                        'Employee ID': record[1],
                        'Employee Name': employee[1],
                        'Department': employee[2],
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
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date.today().replace(day=1))
        with col2:
            end_date = st.date_input("End Date", value=date.today())
        
        if start_date <= end_date:
            # Get attendance data for the range
            attendance_data = db.get_attendance_range(
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if attendance_data:
                # Process data for visualization
                df_data = []
                for record in attendance_data:
                    df_data.append({
                        'ID': record[0],
                        'Employee_ID': record[1], 
                        'Time': record[2],
                        'Date': record[3]
                    })
                df_analytics = pd.DataFrame(df_data)
                df_analytics['Date'] = pd.to_datetime(df_analytics['Date'])
                
                # Daily attendance count
                daily_counts = df_analytics.groupby('Date').size().reset_index()
                daily_counts.columns = ['Date', 'Count']
                
                st.subheader("Daily Attendance Trend")
                st.line_chart(data=daily_counts.set_index('Date'))
                
                # Employee attendance frequency
                employee_counts = df_analytics['Employee_ID'].value_counts()
                st.subheader("Employee Attendance Frequency")
                
                # Get employee names for better display
                employee_names = {}
                for emp_id in employee_counts.index:
                    employee = db.get_employee(emp_id)
                    if employee:
                        employee_names[emp_id] = employee[1]
                
                # Create a readable chart
                chart_data = pd.DataFrame({
                    'Employee': [employee_names.get(emp_id, f"ID: {emp_id}") for emp_id in employee_counts.index],
                    'Days Present': employee_counts.values
                })
                
                st.bar_chart(chart_data.set_index('Employee')['Days Present'])
            else:
                st.info("No attendance data found for the selected date range.")
        else:
            st.error("Start date must be before or equal to end date.")
    
    with tab3:
        st.subheader("Export Attendance Data")
        
        # Export options
        export_type = st.selectbox(
            "Select export type:",
            ["All Attendance Records", "Date Range", "Specific Employee"]
        )
        
        if export_type == "All Attendance Records":
            if st.button("📥 Export All Records"):
                all_records = db.get_all_attendance()
                if all_records:
                    # Create detailed DataFrame
                    export_data = []
                    for record in all_records:
                        employee = db.get_employee(record[1])
                        if employee:
                            export_data.append({
                                'Attendance_ID': record[0],
                                'Employee_ID': record[1],
                                'Employee_Name': employee[1],
                                'Department': employee[2],
                                'Check_in_Time': record[2],
                                'Date': record[3]
                            })
                    
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download All Records",
                        data=csv,
                        file_name=f"attendance_all_records_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No attendance records to export.")
        
        elif export_type == "Date Range":
            col1, col2 = st.columns(2)
            with col1:
                export_start = st.date_input("Export Start Date", value=date.today().replace(day=1))
            with col2:
                export_end = st.date_input("Export End Date", value=date.today())
            
            if st.button("📥 Export Date Range"):
                range_records = db.get_attendance_range(
                    export_start.strftime('%Y-%m-%d'),
                    export_end.strftime('%Y-%m-%d')
                )
                
                if range_records:
                    export_data = []
                    for record in range_records:
                        employee = db.get_employee(record[1])
                        if employee:
                            export_data.append({
                                'Employee_ID': record[1],
                                'Employee_Name': employee[1],
                                'Department': employee[2],
                                'Check_in_Time': record[2],
                                'Date': record[3]
                            })
                    
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Range Records",
                        data=csv,
                        file_name=f"attendance_range_{export_start}_{export_end}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No records found for the selected date range.")

def system_status_page(db):
    st.header("⚙️ System Status")
    
    # System information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Database Statistics")
        
        total_employees = len(db.get_all_employees())
        total_attendance = len(db.get_all_attendance())
        today_attendance = len(db.get_attendance_by_date(date.today().strftime('%Y-%m-%d')))
        
        st.metric("Total Employees", total_employees)
        st.metric("Total Attendance Records", total_attendance)
        st.metric("Today's Attendance", today_attendance)
    
    with col2:
        st.subheader("🔧 System Health")
        
        # Test camera access
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                st.success("✅ Camera: Accessible")
                cap.release()
            else:
                st.error("❌ Camera: Not accessible")
        except:
            st.error("❌ Camera: Error occurred")
        
        # Test database
        try:
            db.get_all_employees()
            st.success("✅ Database: Connected")
        except:
            st.error("❌ Database: Connection error")
        
        # Face recognition library status
        try:
            import face_recognition
            st.success("✅ Face Recognition: Available")
        except:
            st.error("❌ Face Recognition: Not available")
    
    # Database maintenance
    st.subheader("🛠️ Database Maintenance")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clean Old Records", help="Remove attendance records older than 1 year"):
            # This is a placeholder for cleanup functionality
            st.info("Cleanup functionality would be implemented here")
    
    with col2:
        if st.button("📋 Backup Database", help="Create a backup of the current database"):
            st.info("Database backup functionality would be implemented here")

if __name__ == "__main__":
    main()
