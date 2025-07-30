#!/usr/bin/env python3
"""
Photo Upload Trainer - Streamlit app for cloud-based face training
Perfect for cloud environments where camera access is limited
"""

import streamlit as st
import cv2
import numpy as np
import sqlite3
from PIL import Image
import pandas as pd
from datetime import datetime

class PhotoUploadTrainer:
    def __init__(self, db_path='attendance_system.db'):
        self.db_path = db_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def extract_face_features(self, image_bgr):
        """Extract face features from BGR image"""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None
            
        # Get largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        face_roi = cv2.equalizeHist(face_roi)
        
        # Extract features
        hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
        lbp_features = self._calculate_lbp_histogram(face_roi)
        
        # Region features
        regions = []
        h, w = face_roi.shape
        region_h, region_w = h // 5, w // 5
        
        for i in range(5):
            for j in range(5):
                region = face_roi[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                regions.append(np.mean(region))
        
        # Combine features
        combined_features = np.concatenate([
            hist.flatten(),
            lbp_features,
            np.array(regions)
        ])
        
        return combined_features.astype(np.float32)
    
    def _calculate_lbp_histogram(self, image):
        """Calculate Local Binary Pattern histogram"""
        try:
            lbp = np.zeros_like(image)
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    center = image[i, j]
                    code = 0
                    code |= (image[i-1, j-1] >= center) << 7
                    code |= (image[i-1, j] >= center) << 6
                    code |= (image[i-1, j+1] >= center) << 5
                    code |= (image[i, j+1] >= center) << 4
                    code |= (image[i+1, j+1] >= center) << 3
                    code |= (image[i+1, j] >= center) << 2
                    code |= (image[i+1, j-1] >= center) << 1
                    code |= (image[i, j-1] >= center) << 0
                    lbp[i, j] = code
            
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            return hist.astype(np.float32)
        except:
            return np.zeros(256, dtype=np.float32)
    
    def save_to_database(self, employee_id, name, features):
        """Save face features to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            features_binary = features.tobytes()
            
            cursor.execute('''
                UPDATE employees 
                SET face_encoding = ?
                WHERE employee_id = ?
            ''', (features_binary, employee_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Database error: {e}")
            return False

def main():
    st.set_page_config(page_title="Photo Upload Trainer", page_icon="📸")
    
    st.title("📸 Photo Upload Face Trainer")
    st.markdown("**Cloud-based face training for Smart Attendance System**")
    
    trainer = PhotoUploadTrainer()
    
    # Get untrained employees
    conn = sqlite3.connect('attendance_system.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT employee_id, name, department FROM employees 
        WHERE face_encoding IS NULL 
        ORDER BY name
    ''')
    untrained_employees = cursor.fetchall()
    
    cursor.execute('''
        SELECT employee_id, name FROM employees 
        WHERE face_encoding IS NOT NULL 
        ORDER BY name
    ''')
    trained_employees = cursor.fetchall()
    conn.close()
    
    # Display status
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Untrained Employees", len(untrained_employees))
    with col2:
        st.metric("Trained Employees", len(trained_employees))
    
    if not untrained_employees:
        st.success("🎉 All employees have been trained!")
        st.info("You can now use the face recognition system.")
        return
    
    # Training interface
    st.subheader("🔧 Train Employees")
    
    # Select employee
    employee_options = {f"{name} ({emp_id}) - {dept}": (emp_id, name) for emp_id, name, dept in untrained_employees}
    selected = st.selectbox(
        "Select employee to train:",
        options=list(employee_options.keys())
    )
    
    if selected:
        employee_id, employee_name = employee_options[selected]
        
        st.markdown(f"**Training: {employee_name} ({employee_id})**")
        
        # Photo upload
        uploaded_file = st.file_uploader(
            "Upload a clear front-facing photo:",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a high-quality photo with good lighting"
        )
        
        if uploaded_file:
            # Process image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Convert to BGR
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            # Display image
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.image(image, caption="Uploaded Photo", use_container_width=True)
            
            # Extract features
            features = trainer.extract_face_features(image_bgr)
            
            with col_b:
                if features is not None:
                    # Draw face detection
                    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                    faces = trainer.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(25, 25))
                    
                    result_image = image_bgr.copy()
                    for (x, y, w, h) in faces:
                        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(result_image, employee_name, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption="Face Detected ✓", use_container_width=True)
                    
                    # Training button
                    if st.button(f"🎯 Train {employee_name}", type="primary"):
                        success = trainer.save_to_database(employee_id, employee_name, features)
                        
                        if success:
                            st.success(f"✅ Successfully trained {employee_name}!")
                            st.balloons()
                            st.experimental_rerun()
                        else:
                            st.error("❌ Training failed. Please try again.")
                else:
                    st.error("❌ No face detected in the uploaded photo")
                    st.caption("Please upload a clearer photo with a visible face")
    
    # Show training progress
    if trained_employees:
        st.subheader("✅ Trained Employees")
        df = pd.DataFrame(trained_employees, columns=['Employee ID', 'Name'])
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()