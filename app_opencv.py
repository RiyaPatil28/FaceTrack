import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date
import time
from PIL import Image
import io
import base64
import sqlite3
import pickle

# Enhanced database with face detection capabilities
class SmartDatabase:
    def __init__(self, db_path='attendance_system.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create employees table with face data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                face_features BLOB,
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
    
    def add_employee(self, employee_id, name, department, face_features=None):
        """Add a new employee to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize face features if provided
            face_features_blob = pickle.dumps(face_features) if face_features else None
            
            cursor.execute('''
                INSERT INTO employees (employee_id, name, department, face_features)
                VALUES (?, ?, ?, ?)
            ''', (employee_id, name, department, face_features_blob))
            
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

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, frame):
        """Detect faces in frame using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def draw_faces(self, frame, faces):
        """Draw rectangles around detected faces"""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

def main():
    st.set_page_config(
        page_title="Smart Attendance System",
        page_icon="👥",
        layout="wide"
    )
    
    # Initialize database and face detector
    db = SmartDatabase()
    face_detector = FaceDetector()
    
    st.title("🎯 Smart Attendance System")
    st.markdown("AI-powered face detection for automated attendance tracking")
    
    # Add status note
    st.success("📹 **OpenCV Active**: Face detection is now enabled! Face recognition will be added once all dependencies are installed.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📸 Live Attendance", "👤 Employee Management", "📊 Reports & Analytics", "⚙️ System Status"]
    )
    
    if page == "📸 Live Attendance":
        live_attendance_page(db, face_detector)
    elif page == "👤 Employee Management":
        employee_management_page(db)
    elif page == "📊 Reports & Analytics":
        reports_page(db)
    elif page == "⚙️ System Status":
        system_status_page(db)

def live_attendance_page(db, face_detector):
    st.header("📸 Live Attendance Tracking")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed with Face Detection")
        
        # Camera controls
        camera_col1, camera_col2 = st.columns(2)
        with camera_col1:
            start_camera = st.button("🎥 Start Camera", type="primary")
        with camera_col2:
            stop_camera = st.button("⏹️ Stop Camera")
        
        # Camera feed and image upload
        camera_placeholder = st.empty()
        
        # Photo upload for face detection demo
        st.subheader("📷 Upload Photo for Face Detection Demo")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a photo to test face detection"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Convert to BGR for OpenCV
                if len(image_array.shape) == 3:
                    if image_array.shape[2] == 3:  # RGB
                        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    elif image_array.shape[2] == 4:  # RGBA
                        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                    else:
                        image_bgr = image_array
                else:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                
                # Detect faces
                faces = face_detector.detect_faces(image_bgr)
                
                # Draw faces
                image_with_faces = face_detector.draw_faces(image_bgr, faces)
                
                # Convert back to RGB for display
                image_rgb = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)
                
                # Display results
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(image, caption="Original Image", use_column_width=True)
                with col_b:
                    st.image(image_rgb, caption=f"Face Detection Result ({len(faces)} faces detected)", use_column_width=True)
                
                if len(faces) > 0:
                    st.success(f"✅ Successfully detected {len(faces)} face(s) in the uploaded image!")
                    for i, (x, y, w, h) in enumerate(faces):
                        st.info(f"Face {i+1}: Position ({x}, {y}), Size {w}x{h}")
                else:
                    st.warning("⚠️ No faces detected in this image. Try uploading a clearer photo with visible faces.")
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        
        if start_camera:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Cannot access camera. Camera functionality is limited in cloud environments.")
                    st.info("💡 **Tip**: Use the photo upload feature above to test face detection!")
                else:
                    st.success("📹 Camera started successfully!")
                    
                    # Process a few frames for demonstration
                    for i in range(10):  # Limited frames for demo
                        ret, frame = cap.read()
                        if ret:
                            # Detect faces
                            faces = face_detector.detect_faces(frame)
                            
                            # Draw faces
                            frame_with_faces = face_detector.draw_faces(frame, faces)
                            
                            # Convert to RGB for Streamlit
                            frame_rgb = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
                            
                            # Display frame
                            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                            
                            # Add face count info
                            if len(faces) > 0:
                                st.info(f"👥 Detected {len(faces)} face(s) in current frame")
                            
                            time.sleep(0.1)  # Small delay
                    
                    cap.release()
                    st.info("📹 Camera demonstration completed")
                    
            except Exception as e:
                st.error(f"❌ Camera error: {str(e)}")
                st.info("💡 **Alternative**: Use the photo upload feature to test face detection!")
        
        # Manual attendance section
        st.subheader("Manual Attendance")
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
            
            st.info("📸 Photo upload and face recognition training will be enabled once face_recognition library is installed.")
            
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
        st.info("📊 Advanced analytics with trend charts will be available soon.")

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
            st.success(f"✅ OpenCV - OK (v{cv2.__version__})")
        except:
            st.error("❌ OpenCV - Not Available")
        
        try:
            import face_recognition
            st.success("✅ Face Recognition - OK")
        except:
            st.warning("⚠️ Face Recognition - Installing (requires dlib compilation)")
    
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
        
        st.subheader("🎥 Camera Status")
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                st.success("✅ Camera - Available")
                cap.release()
            else:
                st.warning("⚠️ Camera - Not Accessible (Cloud Environment)")
                st.info("💡 Photo upload available as alternative")
        except:
            st.error("❌ Camera - Error")
            st.info("💡 Photo upload available as alternative")

if __name__ == "__main__":
    main()