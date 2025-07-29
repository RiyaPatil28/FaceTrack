#!/usr/bin/env python3
"""
Dataset Trainer for Smart Attendance System
Processes CSV dataset and trains face recognition with real celebrity photos
"""

import pandas as pd
import os
import sqlite3
from PIL import Image
import numpy as np
import cv2
import requests
from datetime import datetime
import time

class DatasetTrainer:
    def __init__(self, db_path='attendance_system.db'):
        self.db_path = db_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def clear_existing_data(self):
        """Clear existing celebrity data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete all CELEB* employees
            cursor.execute("DELETE FROM employees WHERE employee_id LIKE 'CELEB%'")
            cursor.execute("DELETE FROM attendance WHERE employee_id LIKE 'CELEB%'")
            
            conn.commit()
            conn.close()
            print("✓ Cleared existing celebrity data from database")
            
        except Exception as e:
            print(f"Error clearing data: {e}")
    
    def extract_face_features(self, image_path):
        """Extract face features from image file"""
        try:
            if not os.path.exists(image_path):
                return None
                
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
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
            
            # Extract features (simplified version)
            hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            
            # LBP features
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
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
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
    
    def process_dataset(self, csv_path, images_dir=None):
        """Process the dataset CSV and train face recognition"""
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            print(f"Found {len(df)} entries in dataset")
            
            # Get unique celebrities
            celebrities = df['label'].unique()
            print(f"Found {len(celebrities)} unique celebrities: {list(celebrities)}")
            
            # Clear existing data
            self.clear_existing_data()
            
            # Group by celebrity
            trained_count = 0
            failed_count = 0
            
            for celebrity in celebrities:
                celebrity_images = df[df['label'] == celebrity]
                print(f"\nProcessing {celebrity} ({len(celebrity_images)} images)...")
                
                # Try to train with first available image
                trained = False
                for idx, row in celebrity_images.head(5).iterrows():  # Try first 5 images
                    image_file = row['id']
                    
                    # Look for image in attached_assets or specified directory
                    possible_paths = [
                        f"attached_assets/{image_file}",
                        f"attached_assets/celebrity_images/{image_file}",
                        f"celebrity_images/{image_file}",
                        image_file
                    ]
                    
                    if images_dir:
                        possible_paths.insert(0, f"{images_dir}/{image_file}")
                    
                    image_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            image_path = path
                            break
                    
                    if image_path:
                        features = self.extract_face_features(image_path)
                        if features is not None:
                            # Add to database
                            employee_id = f"CELEB{trained_count + 1:03d}"
                            success = self.add_employee_to_db(employee_id, celebrity, features)
                            if success:
                                print(f"  ✓ Trained {celebrity} with {image_file}")
                                trained_count += 1
                                trained = True
                                break
                            else:
                                print(f"  ✗ Failed to add {celebrity} to database")
                        else:
                            print(f"  ⚠ No face detected in {image_file}")
                    else:
                        print(f"  ⚠ Image not found: {image_file}")
                
                if not trained:
                    failed_count += 1
                    print(f"  ✗ Failed to train {celebrity} - no valid images found")
            
            print(f"\n=== Training Complete ===")
            print(f"Successfully trained: {trained_count} celebrities")
            print(f"Failed to train: {failed_count} celebrities")
            
            return trained_count, failed_count
            
        except Exception as e:
            print(f"Error processing dataset: {e}")
            return 0, 0
    
    def add_employee_to_db(self, employee_id, name, face_features):
        """Add employee with face features to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert features to binary for storage
            features_binary = face_features.tobytes()
            
            cursor.execute('''
                INSERT OR REPLACE INTO employees 
                (employee_id, name, department, face_encoding, created_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (employee_id, name, 'Entertainment', features_binary, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error adding employee to database: {e}")
            return False

def main():
    trainer = DatasetTrainer()
    
    # Process the dataset
    csv_path = "attached_assets/Dataset_1753783807479.csv"
    
    if os.path.exists(csv_path):
        print("Starting dataset training...")
        trained, failed = trainer.process_dataset(csv_path)
        
        if trained > 0:
            print(f"\n🎉 Successfully trained {trained} celebrities!")
            print("The face recognition system has been updated with the new dataset.")
        else:
            print("\n❌ No celebrities were successfully trained.")
            print("Please check that the image files are available in the correct directory.")
    else:
        print(f"Dataset file not found: {csv_path}")

if __name__ == "__main__":
    main()