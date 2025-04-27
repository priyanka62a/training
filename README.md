# training
import cv2
import os
import numpy as np

# Directory with saved face images
data_dir = "detected_faces"  
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
faces = []
labels = []

for filename in os.listdir(data_dir):
    img_path = os.path.join(data_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is not None:
        faces.append(cv2.resize(img, (100, 100)))  # Resize images to 100x100
        labels.append(0)  # Assign the same label '0' for all your images

# Train the model
recognizer.train(faces, np.array(labels))
recognizer.save("face_recognizer.yml")
print("Training completed and model saved as 'face_recognizer.yml'")
