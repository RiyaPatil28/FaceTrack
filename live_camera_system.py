#!/usr/bin/env python3
"""
Live Camera System for Real-World Face Recognition Training
Advanced camera detection with multiple fallback options for cloud environments
"""

import streamlit as st
import cv2
import numpy as np
import sqlite3
import time
import threading
from PIL import Image
from datetime import datetime
import os

class LiveCameraTrainer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.db_path = 'attendance_system.db'
        self.camera = None
        self.camera_active = False
        
    def detect_available_cameras(self):
        """Detect all available camera sources"""
        available_cameras = []
        
        # Method 1: Standard camera indices
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append({
                        'id': i,
                        'name': f'Camera {i}',
                        'type': 'USB/Built-in',
                        'resolution': frame.shape[:2]
                    })
                cap.release()
        
        # Method 2: Try different backends
        backends = [
            (cv2.CAP_V4L2, 'V4L2'),
            (cv2.CAP_GSTREAMER, 'GStreamer'),
            (cv2.CAP_FFMPEG, 'FFmpeg'),
            (cv2.CAP_DSHOW, 'DirectShow')
        ]
        
        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append({
                            'id': f'0_{backend}',
                            'name': f'{name} Backend',
                            'type': 'Backend',
                            'resolution': frame.shape[:2]
                        })
                    cap.release()
            except:
                continue
        
        # Method 3: Network cameras (if any)
        network_sources = [
            'http://127.0.0.1:8080/video',  # Common IP camera URL
            'rtsp://127.0.0.1:554/stream',  # RTSP stream
        ]
        
        for source in network_sources:
            try:
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append({
                            'id': source,
                            'name': f'Network Camera',
                            'type': 'Network',
                            'resolution': frame.shape[:2]
                        })
                    cap.release()
            except:
                continue
        
        return available_cameras
    
    def initialize_camera(self, camera_id):
        """Initialize selected camera"""
        try:
            if isinstance(camera_id, str) and '_' in camera_id:
                # Backend-specific camera
                cam_id, backend = camera_id.split('_')
                cam_id = int(cam_id)
                backend = int(backend)
                cap = cv2.VideoCapture(cam_id, backend)
            elif isinstance(camera_id, str):
                # Network camera
                cap = cv2.VideoCapture(camera_id)
            else:
                # Standard camera
                cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                # Set camera properties for better quality
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                return cap
            else:
                return None
                
        except Exception as e:
            st.error(f"Camera initialization error: {e}")
            return None
    
    def run_live_training_interface(self):
        """Main live training interface"""
        st.title("🎥 Live Camera Training System")
        st.markdown("**Real-time face recognition training with camera feed**")
        
        # Camera detection
        with st.spinner("Detecting available cameras..."):
            cameras = self.detect_available_cameras()
        
        if not cameras:
            st.error("❌ No cameras detected in this environment")
            st.info("💡 **Solutions:**")
            st.markdown("1. **Virtual Camera Setup:** Use OBS Virtual Camera or similar software")
            st.markdown("2. **USB Camera:** Connect a USB webcam to the system")
            st.markdown("3. **Network Camera:** Configure an IP camera or smartphone app")
            st.markdown("4. **Browser Access:** Use browser-based camera (if supported)")
            
            # Fallback to photo upload
            st.markdown("---")
            st.subheader("📸 Alternative: Photo Upload Training")
            return self.photo_upload_fallback()
        
        # Camera selection
        st.success(f"📹 Found {len(cameras)} camera source(s)!")
        
        camera_options = {}
        for cam in cameras:
            key = f"{cam['name']} ({cam['type']}) - {cam['resolution'][1]}x{cam['resolution'][0]}"
            camera_options[key] = cam['id']
        
        selected_camera_key = st.selectbox("Select Camera:", list(camera_options.keys()))
        selected_camera_id = camera_options[selected_camera_key]
        
        # Main interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.camera_training_interface(selected_camera_id)
        
        with col2:
            self.training_control_panel()
    
    def camera_training_interface(self, camera_id):
        """Camera training interface"""
        st.subheader("📹 Live Camera Feed")
        
        # Camera controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        with col_ctrl1:
            start_camera = st.button("🔴 Start Camera", type="primary")
        with col_ctrl2:
            stop_camera = st.button("⏹️ Stop Camera")
        with col_ctrl3:
            capture_frame = st.button("📸 Capture Frame")
        
        # Camera feed display
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Session state management
        if start_camera:
            st.session_state.camera_active = True
            st.session_state.selected_camera = camera_id
        
        if stop_camera:
            st.session_state.camera_active = False
            if hasattr(st.session_state, 'camera_object') and st.session_state.camera_object:
                st.session_state.camera_object.release()
        
        # Live camera loop
        if getattr(st.session_state, 'camera_active', False):
            self.run_camera_loop(camera_id, camera_placeholder, status_placeholder, capture_frame)
    
    def run_camera_loop(self, camera_id, camera_placeholder, status_placeholder, capture_frame):
        """Run the live camera loop"""
        # Initialize camera if not already done
        if not hasattr(st.session_state, 'camera_object') or st.session_state.camera_object is None:
            st.session_state.camera_object = self.initialize_camera(camera_id)
        
        cap = st.session_state.camera_object
        
        if cap and cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(50, 50)
                )
                
                # Draw face rectangles
                display_frame = frame.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, 'Face Detected', (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert to RGB for Streamlit
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, caption="Live Camera Feed", use_container_width=True)
                
                # Update status
                if len(faces) > 0:
                    status_placeholder.success(f"👤 {len(faces)} face(s) detected - Ready for training!")
                    
                    # Store current frame for capture
                    st.session_state.current_frame = frame
                    st.session_state.current_faces = faces
                else:
                    status_placeholder.info("🔍 Looking for faces...")
                
                # Handle frame capture
                if capture_frame and len(faces) > 0:
                    st.session_state.captured_frame = frame
                    status_placeholder.success("📸 Frame captured! Ready for training.")
                
            else:
                status_placeholder.error("❌ Camera feed error")
        else:
            status_placeholder.error("❌ Camera not accessible")
    
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
                
                # Training from captured frame
                if hasattr(st.session_state, 'captured_frame'):
                    st.success("📸 Frame ready for training")
                    
                    if st.button(f"🎯 Train {employee_name} with Captured Frame", type="primary"):
                        success = self.train_from_frame(
                            employee_id, 
                            employee_name, 
                            st.session_state.captured_frame
                        )
                        
                        if success:
                            st.success(f"✅ {employee_name} trained successfully!")
                            st.balloons()
                            # Clear captured frame
                            del st.session_state.captured_frame
                            st.experimental_rerun()
                        else:
                            st.error("❌ Training failed")
                
                # Alternative photo upload
                st.markdown("**Or upload photo:**")
                uploaded_file = st.file_uploader(
                    f"Upload photo for {employee_name}",
                    type=['jpg', 'jpeg', 'png'],
                    key=f"upload_{employee_id}"
                )
                
                if uploaded_file:
                    if st.button(f"🎯 Train {employee_name} with Photo", key="train_photo"):
                        success = self.train_from_upload(employee_id, employee_name, uploaded_file)
                        if success:
                            st.success(f"✅ {employee_name} trained successfully!")
                            st.experimental_rerun()
        else:
            st.success("🎉 All employees trained!")
        
        # Training statistics
        self.show_training_stats()
    
    def train_from_frame(self, employee_id, employee_name, frame):
        """Train employee from camera frame"""
        try:
            # Extract face features
            features = self.extract_face_features(frame)
            
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
            st.error(f"Training error: {e}")
            return False
    
    def train_from_upload(self, employee_id, employee_name, uploaded_file):
        """Train employee from uploaded photo"""
        try:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Convert to BGR
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            return self.train_from_frame(employee_id, employee_name, image_bgr)
        except Exception as e:
            st.error(f"Upload training error: {e}")
            return False
    
    def extract_face_features(self, frame):
        """Extract face features from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        face_roi = cv2.equalizeHist(face_roi)
        
        # Create feature vector
        features = face_roi.flatten().astype(np.float32)
        return features
    
    def show_training_stats(self):
        """Show training statistics"""
        st.markdown("---")
        st.subheader("📊 Training Statistics")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM employees')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM employees WHERE face_encoding IS NOT NULL')
        trained = cursor.fetchone()[0]
        
        conn.close()
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total Employees", total)
        with col_s2:
            st.metric("Trained", trained)
        
        if total > 0:
            progress = trained / total
            st.progress(progress)
            st.caption(f"{progress*100:.1f}% Complete")
    
    def photo_upload_fallback(self):
        """Fallback photo upload system"""
        st.subheader("📸 Photo Upload Training")
        
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
            st.success("🎉 All employees trained!")
            return
        
        # Batch training interface
        st.markdown(f"**{len(untrained_employees)} employees need training:**")
        
        for emp_id, name, dept in untrained_employees:
            with st.expander(f"📸 Train {name} ({dept})"):
                uploaded_file = st.file_uploader(
                    f"Upload photo for {name}",
                    type=['jpg', 'jpeg', 'png'],
                    key=f"fallback_{emp_id}"
                )
                
                if uploaded_file:
                    col_img, col_btn = st.columns([1, 1])
                    
                    with col_img:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=f"Photo for {name}", width=200)
                    
                    with col_btn:
                        if st.button(f"🎯 Train {name}", key=f"btn_fallback_{emp_id}"):
                            success = self.train_from_upload(emp_id, name, uploaded_file)
                            if success:
                                st.success(f"✅ {name} trained!")
                                st.experimental_rerun()
                            else:
                                st.error("❌ Training failed")

def main():
    st.set_page_config(page_title="Live Camera Training", page_icon="🎥", layout="wide")
    
    trainer = LiveCameraTrainer()
    trainer.run_live_training_interface()

if __name__ == "__main__":
    main()