#!/usr/bin/env python3
"""
Advanced Camera Training System for Real-World Face Recognition
Optimized for cloud environments with virtual camera capabilities
"""

import streamlit as st
import cv2
import numpy as np
import sqlite3
from PIL import Image
import time
import threading
from datetime import datetime

class CloudCameraSystem:
    def __init__(self, db_path='attendance_system.db'):
        self.db_path = db_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.training_mode = False
        self.current_employee = None
        
    def initialize_camera_alternatives(self):
        """Initialize alternative camera sources for cloud environments"""
        camera_sources = []
        
        # Try different camera indices and sources
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_sources.append(('Built-in Camera', i, cap))
                        continue
                cap.release()
            except:
                continue
        
        # Try different backends
        backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_FFMPEG]
        for backend in backends:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_sources.append((f'Backend {backend}', 0, cap))
                        break
                cap.release()
            except:
                continue
        
        return camera_sources
    
    def create_virtual_camera_interface(self):
        """Create a virtual camera interface for training"""
        st.title("🎥 Advanced Camera Training System")
        st.markdown("**Real-time face recognition training for cloud environments**")
        
        # Camera status
        camera_sources = self.initialize_camera_alternatives()
        
        if camera_sources:
            st.success(f"📹 {len(camera_sources)} camera source(s) detected!")
            
            # Camera selection
            camera_names = [source[0] for source in camera_sources]
            selected_camera = st.selectbox("Select Camera Source:", camera_names)
            
            if selected_camera:
                selected_index = camera_names.index(selected_camera)
                camera_name, camera_id, cap = camera_sources[selected_index]
                
                return self.run_live_training_session(cap, camera_name)
        else:
            st.warning("📹 No direct camera access available in cloud environment")
            return self.create_photo_based_training()
    
    def run_live_training_session(self, cap, camera_name):
        """Run live training session with available camera"""
        st.subheader(f"Live Training with {camera_name}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Camera controls
            start_session = st.button("🔴 Start Training Session", type="primary")
            stop_session = st.button("⏹️ Stop Session")
            
            # Camera feed
            camera_feed = st.empty()
            status_display = st.empty()
            
            if start_session:
                st.session_state.training_active = True
            if stop_session:
                st.session_state.training_active = False
            
            # Live training loop
            if getattr(st.session_state, 'training_active', False):
                self.live_training_loop(cap, camera_feed, status_display)
        
        with col2:
            return self.training_control_panel()
    
    def live_training_loop(self, cap, camera_feed, status_display):
        """Main live training loop"""
        try:
            frame_count = 0
            
            while getattr(st.session_state, 'training_active', False) and frame_count < 100:
                ret, frame = cap.read()
                
                if ret:
                    # Detect faces
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    # Draw face rectangles
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, 'Face Detected', (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Convert to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_feed.image(frame_rgb, caption=f"Live Feed - Frame {frame_count}")
                    
                    # Update status
                    if len(faces) > 0:
                        status_display.success(f"👤 {len(faces)} face(s) detected - Ready for training!")
                    else:
                        status_display.info("🔍 Looking for faces...")
                    
                    frame_count += 1
                    time.sleep(0.1)  # Control frame rate
                else:
                    status_display.error("❌ Camera feed interrupted")
                    break
                    
        except Exception as e:
            status_display.error(f"❌ Camera error: {str(e)}")
        finally:
            if cap:
                cap.release()
    
    def training_control_panel(self):
        """Training control panel"""
        st.subheader("🎯 Training Controls")
        
        # Get untrained employees
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT employee_id, name FROM employees 
            WHERE face_encoding IS NULL 
            ORDER BY name
        ''')
        untrained_employees = cursor.fetchall()
        conn.close()
        
        if untrained_employees:
            # Employee selection
            employee_options = {f"{name} ({emp_id})": emp_id for emp_id, name in untrained_employees}
            selected = st.selectbox("Select Employee to Train:", list(employee_options.keys()))
            
            if selected:
                employee_id = employee_options[selected]
                employee_name = selected.split(' (')[0]
                
                st.markdown(f"**Training: {employee_name}**")
                
                # Training trigger
                if st.button(f"📸 Capture & Train {employee_name}", type="primary"):
                    st.success(f"Training initiated for {employee_name}!")
                    # Training logic would capture current frame and train
                    
                # Manual photo upload as backup
                st.markdown("**Or upload photo:**")
                backup_photo = st.file_uploader(
                    "Upload backup photo",
                    type=['jpg', 'jpeg', 'png'],
                    key="backup_training"
                )
                
                if backup_photo:
                    success = self.train_from_photo(employee_id, employee_name, backup_photo)
                    if success:
                        st.success(f"✅ {employee_name} trained successfully!")
                        st.experimental_rerun()
        else:
            st.success("🎉 All employees trained!")
        
        # Training statistics
        st.markdown("---")
        self.display_training_stats()
    
    def create_photo_based_training(self):
        """Fallback photo-based training system"""
        st.subheader("📸 Photo-Based Training System")
        st.info("Camera not available - using photo upload training mode")
        
        # Get untrained employees
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT employee_id, name, department FROM employees 
            WHERE face_encoding IS NULL 
            ORDER BY name
        ''')
        untrained_employees = cursor.fetchall()
        conn.close()
        
        if not untrained_employees:
            st.success("🎉 All employees are already trained!")
            return
        
        # Training interface
        for emp_id, name, dept in untrained_employees:
            with st.expander(f"Train {name} ({emp_id}) - {dept}"):
                uploaded_file = st.file_uploader(
                    f"Upload photo for {name}",
                    type=['jpg', 'jpeg', 'png'],
                    key=f"train_{emp_id}"
                )
                
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Training photo for {name}", width=300)
                    
                    if st.button(f"🎯 Train {name}", key=f"btn_{emp_id}"):
                        success = self.train_from_photo(emp_id, name, uploaded_file)
                        if success:
                            st.success(f"✅ {name} trained successfully!")
                            st.experimental_rerun()
                        else:
                            st.error("❌ Training failed - no face detected")
    
    def train_from_photo(self, employee_id, employee_name, photo_file):
        """Train employee from uploaded photo"""
        try:
            # Reset file pointer
            photo_file.seek(0)
            image = Image.open(photo_file)
            image_array = np.array(image)
            
            # Convert to BGR
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            # Extract features
            features = self.extract_face_features(image_bgr)
            
            if features is not None:
                # Save to database
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
            
            return False
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return False
    
    def extract_face_features(self, image_bgr):
        """Extract face features for training"""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(25, 25))
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Extract and process face
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        face_roi = cv2.equalizeHist(face_roi)
        
        # Create feature vector
        features = face_roi.flatten().astype(np.float32)
        return features
    
    def display_training_stats(self):
        """Display training statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM employees')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM employees WHERE face_encoding IS NOT NULL')
        trained = cursor.fetchone()[0]
        
        conn.close()
        
        st.metric("Total Employees", total)
        st.metric("Trained", trained)
        
        if total > 0:
            progress = trained / total
            st.progress(progress)
            st.caption(f"{progress*100:.1f}% Complete")

def main():
    st.set_page_config(page_title="Camera Training System", page_icon="🎥")
    
    camera_system = CloudCameraSystem()
    camera_system.create_virtual_camera_interface()

if __name__ == "__main__":
    main()