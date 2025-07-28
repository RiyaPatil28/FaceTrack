import cv2
import numpy as np
import pickle
from typing import List, Tuple, Optional

class SimpleFaceRecognition:
    """
    A simplified face recognition system using OpenCV features
    This provides basic face matching capabilities without requiring dlib
    """
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_faces = {}
        self.is_trained = False
    
    def extract_face_features(self, image):
        """Extract face features using OpenCV's built-in methods"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Calculate histogram features
            hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            
            # Calculate LBP features
            lbp = self._calculate_lbp(face_roi)
            
            # Combine features
            features = np.concatenate([hist.flatten(), lbp.flatten()])
            
            return features
            
        except Exception as e:
            print(f"Error extracting face features: {e}")
            return None
    
    def _calculate_lbp(self, image):
        """Calculate Local Binary Pattern features"""
        try:
            lbp = np.zeros_like(image)
            
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    center = image[i, j]
                    
                    # Calculate LBP value
                    lbp_val = 0
                    lbp_val += (image[i-1, j-1] >= center) << 7
                    lbp_val += (image[i-1, j] >= center) << 6
                    lbp_val += (image[i-1, j+1] >= center) << 5
                    lbp_val += (image[i, j+1] >= center) << 4
                    lbp_val += (image[i+1, j+1] >= center) << 3
                    lbp_val += (image[i+1, j] >= center) << 2
                    lbp_val += (image[i+1, j-1] >= center) << 1
                    lbp_val += (image[i, j-1] >= center) << 0
                    
                    lbp[i, j] = lbp_val
            
            # Calculate histogram of LBP
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            return hist
            
        except Exception as e:
            print(f"Error calculating LBP: {e}")
            return np.zeros((256, 1))
    
    def add_known_face(self, employee_id, name, image):
        """Add a known face to the system"""
        try:
            features = self.extract_face_features(image)
            if features is not None:
                self.known_faces[employee_id] = {
                    'name': name,
                    'features': features
                }
                return True
            return False
        except Exception as e:
            print(f"Error adding known face: {e}")
            return False
    
    def recognize_face(self, image, threshold=0.7):
        """Recognize a face in the given image"""
        try:
            if not self.known_faces:
                return None, 0.0
            
            features = self.extract_face_features(image)
            if features is None:
                return None, 0.0
            
            best_match = None
            best_similarity = 0.0
            
            for employee_id, data in self.known_faces.items():
                known_features = data['features']
                
                # Calculate similarity using correlation
                similarity = self._calculate_similarity(features, known_features)
                
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = employee_id
            
            return best_match, best_similarity
            
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return None, 0.0
    
    def _calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        try:
            # Normalize features
            features1 = features1 / (np.linalg.norm(features1) + 1e-7)
            features2 = features2 / (np.linalg.norm(features2) + 1e-7)
            
            # Calculate cosine similarity
            similarity = np.dot(features1, features2)
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return similarity
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def save_model(self, filepath):
        """Save the face recognition model"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.known_faces, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load the face recognition model"""
        try:
            with open(filepath, 'rb') as f:
                self.known_faces = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_known_employees(self):
        """Get list of known employees"""
        return [(emp_id, data['name']) for emp_id, data in self.known_faces.items()]