import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
import time

# Import our modules
from face_recognition_utils import LiveFaceRecognizer
from database import AttendanceDatabase

def main():
    st.set_page_config(page_title="Camera Recognition", page_icon="📸", layout="wide")
    
    st.title("📸 Camera Recognition System")
    st.markdown("Simple camera interface for instant face recognition and attendance marking")
    
    # Initialize components
    @st.cache_resource
    def init_system():
        recognizer = LiveFaceRecognizer()
        db = AttendanceDatabase()
        return recognizer, db
    
    recognizer, db = init_system()
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Take Photo")
        
        # Simple camera input
        camera_photo = st.camera_input("Take photo for recognition")
        
        if camera_photo is not None:
            # Process immediately
            image = Image.open(camera_photo)
            st.image(image, caption="Photo captured", use_container_width=True)
            
            # Convert for processing
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            # Recognition
            with st.spinner("Processing..."):
                faces, recognized = recognizer.recognize_faces(image_bgr)
            
            # Results
            st.markdown("---")
            st.subheader("🎯 Results")
            
            if len(faces) > 0:
                for i, recognition in enumerate(recognized):
                    if recognition:
                        st.success(f"✅ **{recognition['name']}** recognized!")
                        st.write(f"Employee ID: {recognition['employee_id']}")
                        st.write(f"Confidence: {recognition['confidence']:.1f}%")
                        
                        # Mark attendance
                        if recognition['confidence'] > 70:
                            success = db.mark_attendance(recognition['employee_id'], recognition['confidence'])
                            if success:
                                st.balloons()
                                st.success("🎯 Attendance marked!")
                                current_time = datetime.now().strftime('%H:%M:%S')
                                st.info(f"Time: {current_time}")
                            else:
                                st.warning("Already marked today")
                        else:
                            st.warning(f"Low confidence: {recognition['confidence']:.1f}%")
                    else:
                        st.error("❌ Unknown person")
                        st.info("Train this person in Employee Management")
            else:
                st.warning("No faces detected")
    
    with col2:
        st.markdown("### 📊 Today's Stats")
        
        # Get today's attendance
        conn = sqlite3.connect('attendance_system.db')
        cursor = conn.cursor()
        
        # Today's attendance count
        cursor.execute('''
            SELECT COUNT(*) FROM attendance 
            WHERE DATE(check_in_time) = DATE('now')
        ''')
        today_count = cursor.fetchone()[0]
        
        # Total employees
        cursor.execute('SELECT COUNT(*) FROM employees')
        total_employees = cursor.fetchone()[0]
        
        # Trained employees
        cursor.execute('SELECT COUNT(*) FROM employees WHERE face_encoding IS NOT NULL')
        trained_count = cursor.fetchone()[0]
        
        conn.close()
        
        st.metric("Present Today", today_count)
        st.metric("Total Employees", total_employees)
        st.metric("Trained Faces", trained_count)
        
        if total_employees > 0:
            attendance_rate = (today_count / total_employees) * 100
            st.metric("Attendance Rate", f"{attendance_rate:.1f}%")
        
        # Quick navigation
        st.markdown("### 🔗 Quick Links")
        if st.button("👥 Employee Management", use_container_width=True):
            st.switch_page("app_live.py")
        if st.button("📈 View Reports", use_container_width=True):
            st.switch_page("app_live.py")

if __name__ == "__main__":
    main()