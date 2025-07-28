import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional

class FaceRecognitionUtils:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)
        self.recognition_threshold = 0.6  # Lower is more strict
    
    def detect_faces_opencv(self, frame):
        """Detect faces using OpenCV's Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        return faces
    
    def get_face_encoding(self, image):
        """Get face encoding from an image using face_recognition library"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return None
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if face_encodings:
                return face_encodings[0]  # Return the first face encoding
            
            return None
            
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None
    
    def get_multiple_face_encodings(self, image):
        """Get face encodings for multiple faces in an image"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return [], []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            return face_encodings, face_locations
            
        except Exception as e:
            print(f"Error getting multiple face encodings: {e}")
            return [], []
    
    def compare_faces(self, known_encoding, unknown_encoding, tolerance=None):
        """Compare two face encodings"""
        if tolerance is None:
            tolerance = self.recognition_threshold
        
        try:
            # Calculate face distance
            distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
            
            # Check if faces match
            is_match = distance <= tolerance
            
            # Calculate confidence (inverse of distance, normalized)
            confidence = max(0, (1 - distance) * 100)
            
            return is_match, confidence, distance
            
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return False, 0, 1.0
    
    def find_best_match(self, unknown_encoding, known_encodings_dict, tolerance=None):
        """Find the best match from a dictionary of known encodings"""
        if tolerance is None:
            tolerance = self.recognition_threshold
        
        best_match = None
        best_confidence = 0
        best_distance = 1.0
        
        try:
            for employee_id, known_encoding in known_encodings_dict.items():
                is_match, confidence, distance = self.compare_faces(
                    known_encoding, unknown_encoding, tolerance
                )
                
                if is_match and confidence > best_confidence:
                    best_match = employee_id
                    best_confidence = confidence
                    best_distance = distance
            
            return best_match, best_confidence, best_distance
            
        except Exception as e:
            print(f"Error finding best match: {e}")
            return None, 0, 1.0
    
    def draw_face_box(self, frame, face_location, label="", color=(0, 255, 0)):
        """Draw a bounding box around a detected face"""
        top, right, bottom, left = face_location
        
        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label
        if label:
            # Calculate label size and position
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            thickness = 1
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for label
            cv2.rectangle(
                frame,
                (left, bottom - label_height - 10),
                (left + label_width, bottom),
                color,
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (left + 6, bottom - 6),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return frame
    
    def draw_opencv_face_box(self, frame, face_rect, label="", color=(255, 0, 0)):
        """Draw a bounding box for OpenCV detected faces"""
        x, y, w, h = face_rect
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        if label:
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            thickness = 1
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for label
            cv2.rectangle(
                frame,
                (x, y - label_height - 10),
                (x + label_width, y),
                color,
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x + 6, y - 6),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return frame
    
    def preprocess_frame(self, frame, scale=0.25):
        """Preprocess frame for faster face recognition"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        return rgb_small_frame, scale
    
    def scale_face_locations(self, face_locations, scale_factor):
        """Scale face locations back to original frame size"""
        scaled_locations = []
        for (top, right, bottom, left) in face_locations:
            scaled_locations.append((
                int(top / scale_factor),
                int(right / scale_factor),
                int(bottom / scale_factor),
                int(left / scale_factor)
            ))
        return scaled_locations
    
    def enhance_image_quality(self, image):
        """Enhance image quality for better face recognition"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced_lab = cv2.merge([l, a, b])
            
            # Convert back to BGR
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_image
            
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image
    
    def validate_face_encoding(self, encoding):
        """Validate if a face encoding is valid"""
        if encoding is None:
            return False
        
        if not isinstance(encoding, np.ndarray):
            return False
        
        if encoding.shape != (128,):  # face_recognition produces 128-dimensional encodings
            return False
        
        # Check if encoding contains valid values
        if np.any(np.isnan(encoding)) or np.any(np.isinf(encoding)):
            return False
        
        return True
