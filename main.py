import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure correct Haar Cascade path
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Error: Haar cascade file not found at {cascade_path}")

face_cascade = cv2.CascadeClassifier(cascade_path)

# Get absolute path to images
project_path = os.path.dirname(os.path.abspath(__file__))

# Function to detect and extract a face from an image
def detect_and_extract_face(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        print(f"No faces detected in {image_path}")
        return None

    x, y, w, h = faces[0]
    return gray[y:y+h, x:x+w]

# Function to compare two faces using Mean Squared Error (MSE)
def compare_faces(face1, face2):
    face1_resized = cv2.resize(face1, (100, 100))
    face2_resized = cv2.resize(face2, (100, 100))
    return np.mean((face1_resized - face2_resized) ** 2)

# Function to recognize faces in a given image
def recognize_faces(random_image_path, known_faces):
    if not os.path.exists(random_image_path):
        print(f"Error: File {random_image_path} not found.")
        return

    img = cv2.imread(random_image_path)
    if img is None:
        print(f"Error: Could not read image {random_image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        print(f"No faces detected in {random_image_path}")
        return

    for (x, y, w, h) in faces:
        detected_face = gray[y:y+h, x:x+w]

        best_match_name = "Unknown"
        lowest_difference = float('inf')

        for name, known_face in known_faces.items():
            difference = compare_faces(detected_face, known_face)
            if difference < lowest_difference and difference < 500:
                best_match_name = name
                lowest_difference = difference

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, best_match_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Recognized Faces")
    plt.show()

# Load known faces
known_faces = {
    "Elon Musk": detect_and_extract_face(os.path.join(project_path, "elon_musk.png")),
    "Donald Trump": detect_and_extract_face(os.path.join(project_path, "donald_trump.png")),
    "LeBron James": detect_and_extract_face(os.path.join(project_path, "new_lebron.png"))
}

# Remove None values from known_faces
known_faces = {name: face for name, face in known_faces.items() if face is not None}

# Test with a new image
random_image_path = os.path.join(project_path, "elon_lebron.png")
recognize_faces(random_image_path, known_faces)