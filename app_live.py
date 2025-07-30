import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date
import time
from PIL import Image
import io
import sqlite3
import threading
import queue

# Enhanced database with face recognition capabilities
class LiveDatabase:
    def __init__(self, db_path='attendance_system.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create employees table with face model
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                face_model BLOB,
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
                confidence REAL DEFAULT 0.0,
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
            
            cursor.execute('''
                INSERT INTO employees (employee_id, name, department, face_model)
                VALUES (?, ?, ?, ?)
            ''', (employee_id, name, department, face_features))
            
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
    
    def mark_attendance(self, employee_id, confidence=0.0):
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
                INSERT INTO attendance (employee_id, check_in_time, date, confidence)
                VALUES (?, ?, ?, ?)
            ''', (employee_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), today, confidence))
            
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
                SELECT a.id, a.employee_id, a.check_in_time, a.date, e.name, 
                       COALESCE(a.confidence, 0.0) as confidence
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

class LiveFaceRecognizer:
    """Live face recognition system using OpenCV Haar Cascades and feature matching"""
    
    def __init__(self):
        # Initialize face detector using Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Face recognition parameters
        self.known_faces = {}
        self.recognition_threshold = 0.92  # Much higher threshold for precision
        
        # Initialize feature detector
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
    
    def detect_faces(self, frame):
        """Detect faces using Haar Cascade (fallback method)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # More sensitive detection
            minNeighbors=3,    # Lower threshold for detection
            minSize=(25, 25),  # Even smaller minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def draw_faces(self, frame, faces, labels=None):
        """Draw rectangles around detected faces"""
        for i, (x, y, w, h) in enumerate(faces):
            color = (0, 255, 0) if labels and labels[i] else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            if labels and labels[i]:
                label = f"{labels[i]['name']} ({labels[i]['confidence']:.1f}%)"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def extract_face_features(self, image):
        """Extract face features for recognition using multiple methods"""
        try:
            faces = self.detect_faces(image)
            if len(faces) == 0:
                return None
            
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to larger standard size for better feature extraction
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Apply histogram equalization for better feature consistency
            face_roi = cv2.equalizeHist(face_roi)
            
            # Method 1: Histogram features
            hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            
            # Method 2: LBP (Local Binary Pattern) features
            lbp_features = self._calculate_lbp_histogram(face_roi)
            
            # Method 3: Basic pixel intensity patterns
            # Divide face into regions and calculate mean intensities
            regions = []
            h, w = face_roi.shape
            region_h, region_w = h // 5, w // 5
            
            for i in range(5):
                for j in range(5):
                    region = face_roi[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                    regions.append(np.mean(region))
            
            # Combine all features
            combined_features = np.concatenate([
                hist.flatten(),
                lbp_features,
                np.array(regions)
            ])
            
            return combined_features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _calculate_lbp_histogram(self, image):
        """Calculate Local Binary Pattern histogram"""
        try:
            lbp = np.zeros_like(image)
            
            # Simple LBP calculation
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    center = image[i, j]
                    
                    # 8-neighbor LBP
                    lbp_val = 0
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            lbp_val += (1 << (7-k))
                    
                    lbp[i, j] = lbp_val
            
            # Calculate histogram
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            return hist.flatten()
            
        except Exception as e:
            print(f"Error calculating LBP: {e}")
            return np.zeros(256)
    
    def add_known_face(self, employee_id, name, image):
        """Add a known face to the recognition system with enhanced validation"""
        features = self.extract_face_features(image)
        if features is not None:
            # Validate face quality by checking feature vector properties
            if len(features) > 0 and np.any(features):
                self.known_faces[employee_id] = {
                    'name': name,
                    'features': features,
                    'trained_date': time.time()
                }
                print(f"Successfully trained {name} ({employee_id}) - Feature vector size: {len(features)}")
                return True
            else:
                print(f"Failed to train {name} - Invalid feature vector")
        else:
            print(f"Failed to train {name} - No face detected")
        return False
    
    def recognize_faces(self, frame):
        """Recognize faces in the frame"""
        faces = self.detect_faces(frame)
        recognized = []
        
        for face in faces:
            x, y, w, h = face
            face_roi = frame[y:y+h, x:x+w]
            
            # Simple recognition based on stored features
            features = self.extract_face_features(face_roi)
            if features is not None and self.known_faces:
                best_match = self._find_best_match(features)
                if best_match:
                    recognized.append(best_match)
                else:
                    recognized.append(None)
            else:
                recognized.append(None)
        
        return faces, recognized
    
    def _find_best_match(self, features, threshold=None):
        """Find the best matching known face using multiple similarity metrics"""
        if threshold is None:
            threshold = self.recognition_threshold
            
        best_match = None
        best_score = 0
        
        for emp_id, data in self.known_faces.items():
            try:
                known_features = data['features']
                
                # Method 1: Cosine similarity
                cosine_sim = self._cosine_similarity(features, known_features)
                
                # Method 2: Correlation coefficient
                corr_coeff = np.corrcoef(features, known_features)[0, 1]
                if np.isnan(corr_coeff):
                    corr_coeff = 0
                
                # Method 3: Euclidean distance (inverted and normalized)
                euclidean_dist = np.linalg.norm(features - known_features)
                max_dist = np.linalg.norm(features) + np.linalg.norm(known_features)
                euclidean_sim = 1 - (euclidean_dist / (max_dist + 1e-7))
                
                # Combine scores with weights - focus heavily on cosine similarity for precision
                combined_score = (0.8 * cosine_sim + 0.15 * abs(corr_coeff) + 0.05 * euclidean_sim)
                
                if combined_score > best_score and combined_score > threshold:
                    best_score = combined_score
                    best_match = {
                        'employee_id': emp_id,
                        'name': data['name'],
                        'confidence': combined_score * 100
                    }
                    
            except Exception as e:
                print(f"Error comparing features for {emp_id}: {e}")
                continue
        
        return best_match
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0

def main():
    st.set_page_config(
        page_title="Smart Attendance System - Live",
        page_icon="👥",
        layout="wide"
    )
    
    # Initialize system
    db = LiveDatabase()
    recognizer = LiveFaceRecognizer()
    
    # Load trained celebrities from database
    try:
        conn = sqlite3.connect('attendance_system.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT employee_id, name, face_encoding 
            FROM employees 
            WHERE employee_id LIKE 'CELEB%' AND face_encoding IS NOT NULL
            ORDER BY employee_id
        ''')
        
        celebrities_from_db = cursor.fetchall()
        conn.close()
        
        for emp_id, name, face_encoding_binary in celebrities_from_db:
            try:
                # Convert binary back to numpy array
                face_features = np.frombuffer(face_encoding_binary, dtype=np.float32)
                
                # Add to recognizer
                recognizer.known_faces[emp_id] = {
                    'name': name,
                    'features': face_features,
                    'trained_date': time.time()
                }
                print(f"✓ Loaded {name} ({emp_id}) from database")
                
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
        
        print(f"Face recognition system loaded with {len(recognizer.known_faces)} trained celebrities from dataset")
        
    except Exception as e:
        print(f"Error loading celebrities from database: {e}")
        print("Use the photo upload feature to train employees for face recognition")
    
    # Show training status
    trained_count = len(recognizer.known_faces)
    print(f"Face recognition system initialized with {trained_count} trained employees")
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'last_recognition' not in st.session_state:
        st.session_state.last_recognition = {}
    
    st.title("🎯 Smart Attendance System - Live Recognition")
    st.markdown("Real-time face detection and recognition for automated attendance")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📸 Live Detection", "👤 Employee Management", "📊 Reports", "📱 Mobile Integration", "⚙️ System Status"]
    )
    
    if page == "📸 Live Detection":
        live_detection_page(db, recognizer)
    elif page == "👤 Employee Management":
        employee_management_page(db, recognizer)
    elif page == "📊 Reports":
        reports_page(db)
    elif page == "📱 Mobile Integration":
        from mobile_integration import mobile_integration_page
        mobile_integration_page(db)
    elif page == "⚙️ System Status":
        system_status_page(db)

def live_detection_page(db, recognizer):
    st.header("📸 Live Face Detection & Recognition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed")
        
        # Camera controls
        camera_col1, camera_col2, camera_col3 = st.columns(3)
        with camera_col1:
            if st.button("🎥 Start Live Detection", type="primary"):
                st.session_state.camera_active = True
        with camera_col2:
            if st.button("⏹️ Stop Detection"):
                st.session_state.camera_active = False
        with camera_col3:
            # Show number of trained faces
            trained_count = len(recognizer.known_faces) if hasattr(recognizer, 'known_faces') else 0
            st.metric("Trained Faces", trained_count)
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Live detection loop - Enhanced for cloud environments
        if st.session_state.camera_active:
            try:
                # Try multiple camera indices
                cap = None
                camera_found = False
                
                for camera_index in [0, 1, 2]:
                    try:
                        test_cap = cv2.VideoCapture(camera_index)
                        if test_cap.isOpened():
                            ret, frame = test_cap.read()
                            if ret and frame is not None:
                                cap = test_cap
                                camera_found = True
                                break
                            else:
                                test_cap.release()
                    except:
                        if 'test_cap' in locals():
                            test_cap.release()
                        continue
                
                if not camera_found or cap is None:
                    # Try alternative camera methods for cloud environments
                    status_placeholder.info("🔄 Initializing virtual camera system for cloud environment...")
                    
                    # Create a virtual camera demonstration
                    virtual_camera_demo(camera_placeholder, status_placeholder, recognizer, db)
                    st.session_state.camera_active = False
                else:
                    status_placeholder.success("📹 Live camera detection is active! Move in front of the camera.")
                    
                    # Create a container for real-time updates
                    detection_container = st.container()
                    
                    # Process frames continuously
                    frame_count = 0
                    max_frames = 300  # Run for about 30 seconds at 10fps
                    
                    while st.session_state.camera_active and frame_count < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            status_placeholder.warning("⚠️ Failed to read from camera")
                            break
                        
                        # Process every 3rd frame for better performance
                        if frame_count % 3 == 0:
                            # Detect and recognize faces
                            faces, recognized = recognizer.recognize_faces(frame)
                            
                            # Draw faces with recognition results
                            frame_with_faces = recognizer.draw_faces(frame, faces, recognized)
                            
                            # Mark attendance for recognized faces
                            for i, recognition in enumerate(recognized):
                                if recognition:
                                    emp_id = recognition['employee_id']
                                    current_time = time.time()
                                    last_time = st.session_state.last_recognition.get(emp_id, 0)
                                    
                                    # Prevent duplicate entries within 30 seconds
                                    if current_time - last_time > 30:
                                        success = db.mark_attendance(emp_id, recognition['confidence'])
                                        if success:
                                            st.session_state.last_recognition[emp_id] = current_time
                                            # Show attendance notification
                                            with detection_container:
                                                st.success(f"✅ Attendance marked: {recognition['name']} (Confidence: {recognition['confidence']:.1f}%)")
                            
                            # Display current frame with detection results
                            frame_rgb = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
                            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                            
                            # Show live detection status
                            if len(faces) > 0:
                                if any(recognized):
                                    status_placeholder.success(f"👤 {len(faces)} face(s) detected - Recognition active!")
                                else:
                                    status_placeholder.info(f"👤 {len(faces)} face(s) detected - No matches found")
                            else:
                                status_placeholder.info("🔍 Scanning for faces...")
                        
                        frame_count += 1
                        time.sleep(0.05)  # ~20fps processing
                    
                    cap.release()
                    status_placeholder.info("📹 Live detection session completed")
                    st.session_state.camera_active = False
                    
            except Exception as e:
                status_placeholder.error(f"❌ Error: {str(e)}")
                st.session_state.camera_active = False
        
        # Enhanced photo upload for testing
        st.markdown("---")
        st.subheader("📷 Upload Photo for Face Recognition Testing")
        st.markdown("**Test your trained employees by uploading their photos here:**")
        
        col_upload1, col_upload2 = st.columns([3, 1])
        
        with col_upload1:
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload a photo to test face detection and recognition",
                key="photo_test_upload"
            )
        
        with col_upload2:
            st.markdown("**Quick Test:**")
            st.markdown("Upload photos of:")
            trained_employees = list(recognizer.known_faces.keys()) if hasattr(recognizer, 'known_faces') else []
            if trained_employees:
                for emp_id in trained_employees[:3]:  # Show first 3
                    emp_name = recognizer.known_faces[emp_id]['name']
                    st.markdown(f"• {emp_name}")
            else:
                st.markdown("• Train employees first!")
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Convert to BGR for OpenCV
                if len(image_array.shape) == 3:
                    if image_array.shape[2] == 3:
                        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    else:
                        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                else:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                
                # Detect and recognize faces
                faces, recognized = recognizer.recognize_faces(image_bgr)
                
                # Draw results
                image_with_faces = recognizer.draw_faces(image_bgr, faces, recognized)
                image_rgb = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)
                
                # Display results
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(image, caption="Original Image", use_container_width=True)
                with col_b:
                    st.image(image_rgb, caption=f"Recognition Result ({len(faces)} faces)", use_container_width=True)
                
                # Show detailed recognition results
                st.markdown("### 🎯 Recognition Results:")
                
                # Add training option for unknown faces
                unknown_faces = [i for i, rec in enumerate(recognized) if rec is None]
                if unknown_faces:
                    st.markdown("---")
                    st.subheader("🔄 Train Unknown Face")
                    
                    # Get list of employees without face training
                    conn = sqlite3.connect('attendance_system.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT employee_id, name FROM employees 
                        WHERE face_encoding IS NULL 
                        ORDER BY name
                    ''')
                    untrained_employees = cursor.fetchall()
                    conn.close()
                    
                    if untrained_employees:
                        st.write("Select an employee to train with this photo:")
                        
                        employee_options = {f"{name} ({emp_id})": emp_id for emp_id, name in untrained_employees}
                        selected_employee = st.selectbox(
                            "Choose employee to train:",
                            options=list(employee_options.keys()),
                            key="train_select"
                        )
                        
                        if st.button("🎯 Train Selected Employee", key="train_btn"):
                            if selected_employee:
                                employee_id = employee_options[selected_employee]
                                employee_name = selected_employee.split(' (')[0]
                                
                                # Extract features from the first unknown face
                                first_unknown = unknown_faces[0]
                                face = faces[first_unknown]
                                x, y, w, h = face
                                face_roi = image_bgr[y:y+h, x:x+w]
                                
                                # Train this face
                                success = recognizer.add_known_face(employee_id, employee_name, face_roi)
                                
                                if success:
                                    st.success(f"Successfully trained {employee_name}!")
                                    st.info("The employee will now be recognized in future uploads.")
                                    st.experimental_rerun()
                                else:
                                    st.error("Training failed. Please try with a clearer photo.")
                    else:
                        st.info("All employees have been trained. Add new employees in the Employee Management section.")
                
                if len(faces) == 0:
                    st.warning("⚠️ No faces detected in the uploaded image")
                else:
                    for i, recognition in enumerate(recognized):
                        if recognition:
                            st.success(f"✅ **Recognized Employee #{i+1}:**")
                            st.markdown(f"   - **Name:** {recognition['name']}")
                            st.markdown(f"   - **Employee ID:** {recognition['employee_id']}")
                            st.markdown(f"   - **Confidence:** {recognition['confidence']:.1f}%")
                            
                            # Mark attendance for testing
                            if st.button(f"Mark Attendance for {recognition['name']}", key=f"mark_{i}"):
                                success = db.mark_attendance(recognition['employee_id'], recognition['confidence'])
                                if success:
                                    st.success(f"Attendance marked for {recognition['name']}!")
                                else:
                                    st.info(f"Attendance already marked for {recognition['name']} today")
                        else:
                            st.info(f"👤 **Face #{i+1}:** Unknown person (not in database)")
                            st.markdown("   - This person needs to be trained first")
                        
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
    
    with col2:
        st.subheader("📋 Today's Attendance")
        
        # Get today's attendance
        today = date.today().strftime('%Y-%m-%d')
        today_attendance = db.get_attendance_by_date(today)
        
        if today_attendance:
            for record in today_attendance:
                confidence_text = f" ({record[5]:.1f}%)" if record[5] > 0 else ""
                st.info(f"👤 {record[4]}{confidence_text}\n⏰ {record[2]}")
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

def employee_management_page(db, recognizer):
    st.header("👤 Employee Management")
    
    tab1, tab2 = st.tabs(["➕ Add Employee", "👥 View Employees"])
    
    with tab1:
        st.subheader("Add New Employee or Train Existing Employee Face")
        
        with st.form("add_employee_form"):
            employee_id = st.text_input("Employee ID")
            employee_name = st.text_input("Employee Name")
            department = st.text_input("Department")
            
            st.markdown("**Upload Employee Photo for Face Recognition**")
            uploaded_file = st.file_uploader(
                "Choose image file", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear front-facing photo for face recognition training"
            )
            
            submit_button = st.form_submit_button("Add Employee & Train Face", type="primary")
            
            if submit_button:
                if employee_id and employee_name and uploaded_file:
                    try:
                        # Process uploaded image
                        image = Image.open(uploaded_file)
                        image_array = np.array(image)
                        
                        if len(image_array.shape) == 3:
                            if image_array.shape[2] == 3:
                                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                            else:
                                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                        else:
                            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                        
                        # Train face recognition
                        face_trained = recognizer.add_known_face(employee_id, employee_name, image_bgr)
                        
                        if face_trained:
                            # Try to add new employee or update existing one
                            success = db.add_employee(employee_id, employee_name, department, b'face_trained')
                            
                            if success:
                                st.success(f"✅ Employee {employee_name} added and face trained successfully!")
                                st.rerun()
                            else:
                                # Employee exists, try to update face training
                                try:
                                    conn = sqlite3.connect(db.db_path)
                                    cursor = conn.cursor()
                                    cursor.execute('''
                                        UPDATE employees 
                                        SET name = ?, department = ?, face_model = ?
                                        WHERE employee_id = ?
                                    ''', (employee_name, department, b'face_trained', employee_id))
                                    conn.commit()
                                    conn.close()
                                    
                                    st.success(f"✅ Face training updated for existing employee {employee_name}!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error updating employee: {str(e)}")
                        else:
                            st.error("❌ No face detected in the uploaded image. Please upload a clear photo.")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing: {str(e)}")
                else:
                    st.error("❌ Please fill in all fields and upload an image.")
    
    with tab2:
        st.subheader("All Employees")
        
        employees = db.get_all_employees()
        
        if employees:
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
            
            # Export functionality
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Employees List",
                data=csv,
                file_name=f"employees_list_{date.today()}.csv",
                mime="text/csv"
            )
        else:
            st.info("No employees registered yet.")

def reports_page(db):
    st.header("📊 Reports & Analytics")
    
    # Date selector
    selected_date = st.date_input("Select Date", value=date.today())
    
    # Get attendance for selected date
    attendance_records = db.get_attendance_by_date(selected_date.strftime('%Y-%m-%d'))
    
    if attendance_records:
        df_records = []
        for record in attendance_records:
            df_records.append({
                'Employee ID': record[1],
                'Employee Name': record[4],
                'Check-in Time': record[2],
                'Confidence': f"{record[5]:.1f}%" if record[5] > 0 else "Manual",
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

def virtual_camera_demo(camera_placeholder, status_placeholder, recognizer, db):
    """Virtual camera system for cloud environments with real-time training"""
    
    # Create a simulated camera interface
    status_placeholder.success("📹 Virtual Camera System Active - Real-time Training Mode")
    
    # Training interface
    st.markdown("---")
    st.subheader("🎯 Live Training System")
    st.markdown("**Train employees with real-time face capture:**")
    
    col_train1, col_train2 = st.columns([2, 1])
    
    with col_train1:
        # Get untrained employees
        conn = sqlite3.connect('attendance_system.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT employee_id, name FROM employees 
            WHERE face_encoding IS NULL 
            ORDER BY name
        ''')
        untrained_employees = cursor.fetchall()
        conn.close()
        
        if untrained_employees:
            st.markdown("**Select Employee to Train:**")
            employee_options = {f"{name} ({emp_id})": emp_id for emp_id, name in untrained_employees}
            selected_employee = st.selectbox(
                "Choose employee:",
                options=list(employee_options.keys()),
                key="live_train_select"
            )
            
            # Real camera capture using browser
            st.markdown("**📹 Live Camera Capture:**")
            camera_image = st.camera_input("Take photo with camera for training")
            
            # Fallback upload option
            st.markdown("**Or upload photo:**")
            uploaded_photo = st.file_uploader(
                "Upload employee photo if camera not available",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear front-facing photo for training",
                key="live_training_upload"
            )
            
            # Handle both camera capture and file upload
            photo_source = camera_image if camera_image is not None else uploaded_photo
            
            if photo_source and selected_employee:
                employee_id = employee_options[selected_employee]
                employee_name = selected_employee.split(' (')[0]
                
                # Process the image (camera or upload)
                image = Image.open(photo_source)
                image_array = np.array(image)
                
                if len(image_array.shape) == 3:
                    if image_array.shape[2] == 3:
                        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    else:
                        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                else:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                
                # Display live preview with source info
                source_type = "📹 Camera Capture" if camera_image is not None else "📁 File Upload"
                camera_placeholder.image(image, caption=f"{source_type} - Training: {employee_name}", use_container_width=True)
                
                # Auto-train button
                if st.button(f"🔴 TRAIN {employee_name}", type="primary", key="auto_train"):
                    success = recognizer.add_known_face(employee_id, employee_name, image_bgr)
                    
                    if success:
                        st.success(f"✅ {employee_name} trained successfully!")
                        st.balloons()
                        
                        # Update database
                        try:
                            conn = sqlite3.connect('attendance_system.db')
                            cursor = conn.cursor()
                            cursor.execute('''
                                UPDATE employees 
                                SET face_encoding = ?
                                WHERE employee_id = ?
                            ''', (b'trained', employee_id))
                            conn.commit()
                            conn.close()
                        except:
                            pass
                        
                        st.experimental_rerun()
                    else:
                        st.error("❌ Training failed. No clear face detected.")
        else:
            st.info("🎉 All employees are trained!")
            st.markdown("**Add new employees first:**")
            
            # Quick employee addition interface
            with st.expander("➕ Add New Employee"):
                new_emp_id = st.text_input("Employee ID", placeholder="EMP001")
                new_emp_name = st.text_input("Employee Name", placeholder="John Doe") 
                new_emp_dept = st.text_input("Department", placeholder="Engineering")
                
                if st.button("Add Employee", type="primary"):
                    if new_emp_id and new_emp_name and new_emp_dept:
                        try:
                            conn = sqlite3.connect('attendance_system.db')
                            cursor = conn.cursor()
                            cursor.execute('''
                                INSERT INTO employees (employee_id, name, department, created_date)
                                VALUES (?, ?, ?, ?)
                            ''', (new_emp_id, new_emp_name, new_emp_dept, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                            conn.commit()
                            conn.close()
                            st.success(f"Added {new_emp_name}!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        st.error("Please fill all fields")
            
            st.info("System ready for live recognition.")
    
    with col_train2:
        st.markdown("**Training Status:**")
        
        # Show training progress
        conn = sqlite3.connect('attendance_system.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM employees')
        total_employees = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM employees WHERE face_encoding IS NOT NULL')
        trained_employees = cursor.fetchone()[0]
        
        conn.close()
        
        st.metric("Total Employees", total_employees)
        st.metric("Trained", trained_employees)
        
        if total_employees > 0:
            progress = trained_employees / total_employees
            st.progress(progress)
            st.caption(f"{progress*100:.1f}% Complete")
        
        # Live recognition test with camera
        st.markdown("**Test Recognition:**")
        
        # Camera option for testing
        test_camera = st.camera_input("Take photo to test recognition")
        
        # Upload option as fallback
        test_upload = st.file_uploader(
            "Or upload test photo",
            type=['jpg', 'jpeg', 'png'],
            key="test_recognition"
        )
        
        # Process either camera or upload
        test_source = test_camera if test_camera is not None else test_upload
        
        if test_source:
            test_image = Image.open(test_source)
            test_array = np.array(test_image)
            
            if len(test_array.shape) == 3:
                if test_array.shape[2] == 3:
                    test_bgr = cv2.cvtColor(test_array, cv2.COLOR_RGB2BGR)
                else:
                    test_bgr = cv2.cvtColor(test_array, cv2.COLOR_RGBA2BGR)
            else:
                test_bgr = cv2.cvtColor(test_array, cv2.COLOR_GRAY2BGR)
            
            # Recognize faces
            faces, recognized = recognizer.recognize_faces(test_bgr)
            
            if len(faces) > 0:
                for i, recognition in enumerate(recognized):
                    if recognition:
                        st.success(f"✅ {recognition['name']}")
                        st.caption(f"Confidence: {recognition['confidence']:.1f}%")
                        
                        # Auto-mark attendance
                        if recognition['confidence'] > 85:
                            success = db.mark_attendance(recognition['employee_id'], recognition['confidence'])
                            if success:
                                st.info("Attendance marked!")
                    else:
                        st.warning("Unknown person")
            else:
                st.warning("No faces detected")

def system_status_page(db):
    st.header("⚙️ System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 System Components")
        
        try:
            import cv2
            st.success(f"✅ OpenCV - OK (v{cv2.__version__})")
        except:
            st.error("❌ OpenCV - Not Available")
        
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
    
    with col2:
        st.subheader("🗄️ Database & Recognition Status")
        
        try:
            employees = db.get_all_employees()
            st.success("✅ Database Connection - OK")
            st.info(f"📊 Total Employees: {len(employees)}")
            
            today = date.today().strftime('%Y-%m-%d')
            today_attendance = db.get_attendance_by_date(today)
            st.info(f"📅 Today's Attendance: {len(today_attendance)}")
            
            known_faces = len(getattr(recognizer, 'known_faces', {}))
            st.info(f"🧠 Trained Faces: {known_faces}")
            
        except Exception as e:
            st.error(f"❌ System Error: {str(e)}")

if __name__ == "__main__":
    main()