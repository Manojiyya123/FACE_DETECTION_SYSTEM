import cv2
import os
import numpy as np
import time
from datetime import datetime

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create recognizer (try FisherFace first, fallback to LBPH)
try:
    recognizer = cv2.face.FisherFaceRecognizer_create()
    print("Using FisherFace recognizer for better accuracy")
except:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Using LBPH recognizer")

# Configuration
DATA_DIR = 'face_data'
MODEL_FILE = 'face_model.yml'
LABEL_FILE = 'label_mapping.txt'
MIN_FACE_SIZE = 150  # Minimum face size in pixels
SAMPLE_COUNT = 30    # Number of samples per person
CONFIDENCE_THRESHOLD = 70  # Minimum confidence to accept recognition

def setup_directories():
    """Create required directories"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def collect_samples(name):
    """Capture high-quality face samples"""
    user_dir = os.path.join(DATA_DIR, name)
    os.makedirs(user_dir, exist_ok=True)
    
    # Clear existing samples
    for f in os.listdir(user_dir):
        os.remove(os.path.join(user_dir, f))
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    count = 0
    last_capture = time.time()
    
    while count < SAMPLE_COUNT:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Capture if exactly one face detected and enough time passed
        if len(faces) == 1 and time.time() - last_capture > 0.3:
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            
            # Validate face quality
            if 0.8 < w/h < 1.2:  # Reasonable aspect ratio
                # Preprocess and save
                face_img = cv2.equalizeHist(face_img)
                face_img = cv2.resize(face_img, (200, 200))
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(user_dir, f"{count}_{timestamp}.jpg"), face_img)
                count += 1
                last_capture = time.time()
                print(f"Saved sample {count}/{SAMPLE_COUNT}")
        
        # Display status
        cv2.putText(frame, f"Collecting {name}: {count}/{SAMPLE_COUNT}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Move your head slightly", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Collecting Samples', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished collecting {count} samples for {name}")

def train_model():
    """Train the recognition model with augmentation and validation"""
    faces = []
    labels = []
    label_ids = {}
    current_id = 0
    
    # Get all person directories
    for person_name in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        label_ids[person_name] = current_id
        print(f"Processing {person_name} (label {current_id})...")
        
        # Process each sample
        for img_file in os.listdir(person_dir):
            if not img_file.endswith('.jpg'):
                continue
                
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Original image
            faces.append(img)
            labels.append(current_id)
            
            # Data augmentation
            # 1. Small rotations
            for angle in [-10, 0, 10]:
                M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
                rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                faces.append(rotated)
                labels.append(current_id)
            
            # 2. Blurred version
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            faces.append(blurred)
            labels.append(current_id)
            
            # 3. Flipped version
            flipped = cv2.flip(img, 1)
            faces.append(flipped)
            labels.append(current_id)
        
        current_id += 1

    # Validation checks
    if len(faces) == 0:
        print("Error: No training data found. Please register at least two people with samples.")
        return

    unique_labels = set(labels)
    print(f"\nTotal samples collected: {len(faces)}")
    print(f"Number of unique people (classes): {len(unique_labels)}")
    
    if len(unique_labels) < 2:
        print("Error: At least two distinct people are required to train the model.")
        print("Please register one more person using option 1.")
        return

    # Train and save model
    print("Training model. Please wait...")
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    
    # Save label mapping
    with open(LABEL_FILE, 'w') as f:
        for name, id in label_ids.items():
            f.write(f"{name},{id}\n")
    
    print(f"Training complete. Model saved to {MODEL_FILE}")

def recognize_faces():
    """Recognize faces with confidence checking"""
    # Load label mapping
    try:
        with open(LABEL_FILE, 'r') as f:
            label_map = {}
            for line in f:
                name, id = line.strip().split(',')
                label_map[int(id)] = name
    except FileNotFoundError:
        print("Error: No trained model found. Train first.")
        return
    
    # Load recognizer
    try:
        recognizer.read(MODEL_FILE)
    except:
        print("Error: Couldn't load recognizer. Train first.")
        return
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)
            
            # Get prediction
            id, confidence = recognizer.predict(face_roi)
            confidence_percent = max(0, 100 - confidence)
            
            # Determine recognition result
            if confidence_percent >= CONFIDENCE_THRESHOLD:
                name = label_map.get(id, "Unknown")
                color = (0, 255, 0)  # Green
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red
            
            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"{confidence_percent:.1f}%", (x, y+h+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def list_people():
    """List all registered people"""
    if not os.path.exists(DATA_DIR):
        print("No registered people yet")
        return
    
    people = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if not people:
        print("No registered people yet")
    else:
        print("\nRegistered People:")
        for i, name in enumerate(people, 1):
            samples = len([f for f in os.listdir(os.path.join(DATA_DIR, name)) if f.endswith('.jpg')])
            print(f"{i}. {name} ({samples} samples)")

def delete_person(name):
    """Delete a registered person"""
    person_dir = os.path.join(DATA_DIR, name)
    if os.path.exists(person_dir):
        for f in os.listdir(person_dir):
            os.remove(os.path.join(person_dir, f))
        os.rmdir(person_dir)
        print(f"Deleted {name}")
        
        # Remove from model if exists
        if os.path.exists(MODEL_FILE):
            print("Note: You should retrain the model")
    else:
        print(f"Error: {name} not found")

def main_menu():
    """Main program interface"""
    setup_directories()
    
    while True:
        print("\n=== Face Recognition System ===")
        print("1. Register new person")
        print("2. Train model")
        print("3. Recognize faces")
        print("4. List registered people")
        print("5. Delete person")
        print("6. Exit")
        
        choice = input("Select option: ").strip()
        
        if choice == '1':
            name = input("Enter person's name: ").strip()
            if name:
                collect_samples(name)
            else:
                print("Error: Name cannot be empty")
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            list_people()
        elif choice == '5':
            name = input("Enter name to delete: ").strip()
            if name:
                delete_person(name)
            else:
                print("Error: Name cannot be empty")
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main_menu()