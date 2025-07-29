#!/usr/bin/env python3
"""
Script to automatically train celebrity faces for the attendance system
"""

import cv2
import numpy as np
from PIL import Image
import sys
import os

# Add current directory to path to import our modules
sys.path.append('.')

from app_live import LiveFaceRecognizer, LiveDatabase

def train_celebrity_faces():
    """Train all celebrity faces with their respective photos"""
    
    # Initialize the recognition system
    recognizer = LiveFaceRecognizer()
    db = LiveDatabase()
    
    # Celebrity mapping: employee_id -> (name, image_filename)
    celebrities = {
        'CELEB001': ('Emma Watson', 'attached_assets/image_1753715616096.png'),
        'CELEB002': ('Emma Stone', 'attached_assets/image_1753715656090.png'),
        'CELEB003': ('Dua Lipa', 'attached_assets/image_1753715683140.png'),
        'CELEB004': ('Harry Styles', 'attached_assets/image_1753715722850.png'),
        'CELEB006': ('Selena Gomez', 'attached_assets/image_1753715817958.png'),
    }
    
    trained_count = 0
    
    print("Starting celebrity face training...")
    print("=" * 50)
    
    for emp_id, (name, image_path) in celebrities.items():
        try:
            print(f"Training {name} ({emp_id})...")
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"  ❌ Image not found: {image_path}")
                continue
            
            # Load and process image
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Convert to BGR for OpenCV
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                elif image_array.shape[2] == 4:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                else:
                    print(f"  ❌ Unsupported image format for {name}")
                    continue
            else:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            # Train the face
            success = recognizer.add_known_face(emp_id, name, image_bgr)
            
            if success:
                print(f"  ✅ {name} trained successfully!")
                trained_count += 1
            else:
                print(f"  ❌ Failed to detect face in {name}'s photo")
                
        except Exception as e:
            print(f"  ❌ Error training {name}: {str(e)}")
            continue
    
    print("=" * 50)
    print(f"Training completed: {trained_count}/{len(celebrities)} celebrities trained")
    
    if trained_count > 0:
        print("\n🎯 Trained employees:")
        for emp_id in recognizer.known_faces.keys():
            emp_name = recognizer.known_faces[emp_id]['name']
            print(f"  - {emp_id}: {emp_name}")
        
        print("\n✅ Face recognition system is ready!")
        print("You can now test recognition by uploading different photos of these celebrities.")
    else:
        print("\n❌ No faces were trained. Please check the image files.")
    
    return trained_count

if __name__ == "__main__":
    train_celebrity_faces()