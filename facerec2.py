import cv2
import os
import numpy as np
import pickle
import shutil

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists('dataset'):
    os.makedirs('dataset')
import time

def start_detection_and_redirect():
    recognizer.read('trainer.yml')
    with open("labels.pickle", "rb") as f:
        label_ids = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "❌ Cannot access webcam"

    recognized_name = None
    start_time = time.time()
    TIMEOUT = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id_, confidence = recognizer.predict(roi_gray)
            accuracy = round(100 - confidence)

            if accuracy > 50:
                name = list(label_ids.keys())[list(label_ids.values()).index(id_)]
                recognized_name = name
                label = f"{name}: {accuracy}%"
                color = (0, 255, 0)

                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                break  # face found, exit

        cv2.imwrite('static/detected.jpg', frame)

        if recognized_name or (time.time() - start_time > TIMEOUT):
            break

    cap.release()
    return recognized_name


def collect_faces(name):
    user_dir = f'dataset/{name}'
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < 30:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(f'{user_dir}/{count}.jpg', face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Saved: {count}/30", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Collecting Faces (Press q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def train_model():
    faces = []
    labels = []
    label_ids = {}
    current_id = 0
    
    for root, dirs, files in os.walk('dataset'):
        for dir_name in dirs:
            label_ids[dir_name] = current_id
            subject_dir = os.path.join(root, dir_name)
            
            for filename in os.listdir(subject_dir):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(subject_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    faces.append(img)
                    labels.append(current_id)
            
            current_id += 1
    
    recognizer.train(faces, np.array(labels))
    recognizer.save('trainer.yml')
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    print(f"✅ Trained on {len(faces)} images.")


def recognize_faces():
    try:
        recognizer.read('trainer.yml')
        with open("labels.pickle", "rb") as f:
            label_ids = pickle.load(f)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                id_, confidence = recognizer.predict(roi_gray)

                if confidence < 100:
                    name = list(label_ids.keys())[list(label_ids.values()).index(id_)]
                    text = f"{name} {round(100 - confidence)}%"
                else:
                    text = "Unknown"

                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Face Recognition (Press q to quit)', frame)
            
            # Save the current frame for Flask if needed
            if not os.path.exists('static'):
                os.makedirs('static')
            cv2.imwrite('static/current_recognition.jpg', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Recognition error: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
def list_faces():
    if not os.path.exists("dataset"):
        return []
    return [name for name in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", name))]

def delete_face(name):
    path = f"dataset/{name}"
    
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"✅ Deleted data for '{name}'")
    else:
        print(f"❌ No data found for '{name}'")

if __name__ == "__main__":
    while True:
        print("\n=== Face Recognition Menu ===")
        print("1. Register New Face")
        print("2. Train Model")
        print("3. Recognize Faces")
        print("4. List Faces")
        print("5. Delete Face")
        print("6. Exit")
        choice = input("Select option (1-6): ")
        
        if choice == '1':
            name = input("Enter your name: ")
            collect_faces(name)
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            list_faces()
        elif choice == '5':
            delete_face()
        elif choice == '6':
            print("👋 Exiting program.")
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")
