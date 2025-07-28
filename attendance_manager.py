import cv2
import numpy as np
from datetime import datetime, time
from typing import List, Dict, Tuple, Optional
import threading
import time as time_module

from database import Database
from face_recognition_utils import FaceRecognitionUtils

class AttendanceManager:
    def __init__(self, database: Database, face_utils: FaceRecognitionUtils):
        self.db = database
        self.face_utils = face_utils
        self.known_faces_cache = {}
        self.last_cache_update = 0
        self.cache_refresh_interval = 300  # 5 minutes
        self.processing_lock = threading.Lock()
        
        # Load known faces into cache
        self.refresh_known_faces_cache()
    
    def refresh_known_faces_cache(self):
        """Refresh the cache of known face encodings"""
        try:
            with self.processing_lock:
                employees = self.db.get_all_employees_with_encodings()
                self.known_faces_cache = {}
                
                for employee in employees:
                    employee_id = employee[0]
                    face_encoding = employee[4]
                    
                    if self.face_utils.validate_face_encoding(face_encoding):
                        self.known_faces_cache[employee_id] = {
                            'encoding': face_encoding,
                            'name': employee[1],
                            'department': employee[2]
                        }
                
                self.last_cache_update = time_module.time()
                print(f"Cache refreshed with {len(self.known_faces_cache)} employees")
                
        except Exception as e:
            print(f"Error refreshing cache: {e}")
    
    def check_cache_freshness(self):
        """Check if cache needs to be refreshed"""
        current_time = time_module.time()
        if current_time - self.last_cache_update > self.cache_refresh_interval:
            self.refresh_known_faces_cache()
    
    def process_frame(self, frame):
        """Process a frame and return detected/recognized faces"""
        try:
            # Check if cache needs refresh
            self.check_cache_freshness()
            
            if not self.known_faces_cache:
                # No known faces to compare against
                return self.draw_detection_only(frame)
            
            # Enhance image quality
            enhanced_frame = self.face_utils.enhance_image_quality(frame)
            
            # Preprocess frame for faster recognition
            small_frame, scale = self.face_utils.preprocess_frame(enhanced_frame, scale=0.5)
            
            # Get face encodings from the frame
            face_encodings, face_locations = self.face_utils.get_multiple_face_encodings(small_frame)
            
            # Scale face locations back to original size
            face_locations = self.face_utils.scale_face_locations(face_locations, scale)
            
            # Process frame with recognition results
            processed_frame, recognized_faces = self.process_recognition_results(
                frame, face_encodings, face_locations
            )
            
            return processed_frame, recognized_faces
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return self.draw_detection_only(frame)
    
    def draw_detection_only(self, frame):
        """Draw face detection boxes when no known faces are available"""
        try:
            # Use OpenCV for basic face detection
            faces = self.face_utils.detect_faces_opencv(frame)
            
            for (x, y, w, h) in faces:
                self.face_utils.draw_opencv_face_box(
                    frame, (x, y, w, h), "Unknown Face", (0, 0, 255)
                )
            
            # Add status text
            cv2.putText(
                frame,
                f"Faces detected: {len(faces)} (No known faces registered)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            return frame, []
            
        except Exception as e:
            print(f"Error in detection-only mode: {e}")
            return frame, []
    
    def process_recognition_results(self, frame, face_encodings, face_locations):
        """Process face recognition results and draw annotations"""
        recognized_faces = []
        
        try:
            # Prepare known encodings for comparison
            known_encodings = {emp_id: data['encoding'] 
                             for emp_id, data in self.known_faces_cache.items()}
            
            for i, face_encoding in enumerate(face_encodings):
                face_location = face_locations[i]
                
                # Find best match
                best_match, confidence, distance = self.face_utils.find_best_match(
                    face_encoding, known_encodings
                )
                
                if best_match:
                    # Recognized face
                    employee_data = self.known_faces_cache[best_match]
                    label = f"{employee_data['name']} ({confidence:.1f}%)"
                    color = (0, 255, 0)  # Green for recognized
                    
                    # Check if already marked today
                    already_marked = self.db.is_attendance_marked_today(best_match)
                    if already_marked:
                        label += " [MARKED]"
                        color = (0, 165, 255)  # Orange for already marked
                    
                    recognized_faces.append({
                        'employee_id': best_match,
                        'name': employee_data['name'],
                        'confidence': confidence,
                        'distance': distance,
                        'already_marked': already_marked
                    })
                    
                else:
                    # Unknown face
                    label = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                
                # Draw face box with label
                self.face_utils.draw_face_box(frame, face_location, label, color)
            
            # Add frame information
            self.add_frame_info(frame, len(face_encodings), len(recognized_faces))
            
            return frame, recognized_faces
            
        except Exception as e:
            print(f"Error processing recognition results: {e}")
            return frame, []
    
    def add_frame_info(self, frame, total_faces, recognized_faces):
        """Add information overlay to the frame"""
        try:
            # Current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Information text
            info_lines = [
                f"Time: {current_time}",
                f"Faces: {total_faces} | Recognized: {recognized_faces}",
                f"Known employees: {len(self.known_faces_cache)}"
            ]
            
            # Draw background rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw text lines
            for i, line in enumerate(info_lines):
                y_pos = 35 + (i * 25)
                cv2.putText(
                    frame,
                    line,
                    (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
        except Exception as e:
            print(f"Error adding frame info: {e}")
    
    def mark_attendance(self, employee_id):
        """Mark attendance for an employee if not already marked today"""
        try:
            # Check if already marked today
            if self.db.is_attendance_marked_today(employee_id):
                return False
            
            # Mark attendance
            success = self.db.mark_attendance(employee_id)
            
            if success:
                print(f"Attendance marked for employee: {employee_id}")
                return True
            else:
                print(f"Failed to mark attendance for employee: {employee_id}")
                return False
                
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False
    
    def get_today_attendance(self):
        """Get today's attendance records"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            return self.db.get_attendance_by_date(today)
            
        except Exception as e:
            print(f"Error getting today's attendance: {e}")
            return []
    
    def get_attendance_summary(self, start_date=None, end_date=None):
        """Get attendance summary for a date range"""
        try:
            if start_date and end_date:
                records = self.db.get_attendance_range(start_date, end_date)
            else:
                records = self.get_today_attendance()
            
            # Process records to get summary
            summary = {
                'total_records': len(records),
                'unique_employees': len(set(record[1] for record in records)),
                'records_by_employee': {}
            }
            
            for record in records:
                employee_id = record[1]
                if employee_id not in summary['records_by_employee']:
                    employee_data = self.known_faces_cache.get(employee_id, {})
                    summary['records_by_employee'][employee_id] = {
                        'name': employee_data.get('name', f'ID: {employee_id}'),
                        'department': employee_data.get('department', 'Unknown'),
                        'count': 0,
                        'dates': []
                    }
                
                summary['records_by_employee'][employee_id]['count'] += 1
                summary['records_by_employee'][employee_id]['dates'].append(record[3])
            
            return summary
            
        except Exception as e:
            print(f"Error getting attendance summary: {e}")
            return {'total_records': 0, 'unique_employees': 0, 'records_by_employee': {}}
    
    def validate_system_status(self):
        """Validate system components status"""
        status = {
            'database': False,
            'known_faces': False,
            'face_recognition': False,
            'opencv': False
        }
        
        try:
            # Test database
            self.db.get_all_employees()
            status['database'] = True
        except:
            pass
        
        try:
            # Test known faces cache
            if self.known_faces_cache:
                status['known_faces'] = True
        except:
            pass
        
        try:
            # Test face recognition
            import face_recognition
            status['face_recognition'] = True
        except:
            pass
        
        try:
            # Test OpenCV
            test_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not test_cascade.empty():
                status['opencv'] = True
        except:
            pass
        
        return status
    
    def cleanup_old_records(self, days_to_keep=365):
        """Clean up attendance records older than specified days"""
        try:
            import sqlite3
            from datetime import timedelta
            
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
            
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM attendance WHERE date < ?', (cutoff_date,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"Cleaned up {deleted_count} old attendance records")
            return deleted_count
            
        except Exception as e:
            print(f"Error cleaning up old records: {e}")
            return 0
    
    def export_employee_data(self):
        """Export employee data for backup"""
        try:
            employees = self.db.get_all_employees()
            
            export_data = []
            for employee in employees:
                export_data.append({
                    'employee_id': employee[0],
                    'name': employee[1],
                    'department': employee[2],
                    'created_date': employee[3]
                    # Note: Face encodings are not exported for security
                })
            
            return export_data
            
        except Exception as e:
            print(f"Error exporting employee data: {e}")
            return []
