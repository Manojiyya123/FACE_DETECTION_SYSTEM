from flask import Flask, render_template, request, redirect, jsonify
from facerec2 import collect_faces, train_model, recognize_faces, list_faces, delete_face
import threading
import os  # Add this import
from facerec2 import start_detection_and_redirect

training_done = False
app = Flask(__name__)

@app.route('/')
def home():
    try:
        people = list_faces()  # Get list from facerec2.py
    except Exception as e:
        people = []  # If there's an error, use an empty list
    return render_template('index.html', people=people)



@app.route('/collect_faces', methods=['POST'])
def collect():
    name = request.form['person_name']
    try:
        # Run collection in thread
        thread = threading.Thread(target=collect_faces, args=(name,))
        thread.start()
        return jsonify({"status": "collection_started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train():
    # Run in a thread to avoid blocking
    def train_and_notify():
        global training_done
        train_model()  # This will print the number of images
        training_done = True
    
    thread = threading.Thread(target=train_and_notify)
    thread.start()
    return jsonify({"status": "training_started"})

@app.route('/train_status', methods=['GET'])
def train_status():
    return jsonify({"done": training_done})

@app.route('/get_training_stats', methods=['GET'])
@app.route('/get_training_stats', methods=['GET'])
def get_training_stats():
    try:
        faces_count = 0
        if os.path.exists('dataset'):
            for root, dirs, files in os.walk('dataset'):
                for file in files:
                    if file.endswith('.jpg'):
                        faces_count += 1
        return jsonify({"num_images": faces_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/recognize', methods=['POST'])
def recognize():
    # Run in a thread to avoid blocking
    thread = threading.Thread(target=recognize_faces)
    thread.start()
    return redirect('/')

@app.route('/list_faces', methods=['GET'])
def list_people():
    people = list_faces()
    return jsonify({"people": people})

@app.route('/start_detection', methods=['GET'])
def start_detection():
    name = start_detection_and_redirect()
    if name:
        return redirect(f"/welcome/{name}")
    else:
        return "<h3>No face recognized with enough confidence. Please try again.</h3>"
@app.route('/welcome/<name>')
def welcome(name):
    return f"<h2>🎉 Welcome, {name}!</h2>"


@app.route('/delete_face', methods=['POST'])
def delete():
    name = request.form['person_name']
    try:
        delete_face(name)
        return jsonify({"success": True, "message": f"Deleted {name}"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400
if __name__ == '__main__':
    app.run(debug=True)