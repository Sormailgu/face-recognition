import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
from scipy.spatial.distance import cosine
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for React app

# Directory containing face images
PEOPLE_DIR = "../dataset"
EMBEDDINGS_FILE = "embeddings.pkl"

# Load or compute face embeddings
def load_embeddings():
    # Check if embeddings.pkl exists and is non-empty
    if os.path.exists(EMBEDDINGS_FILE) and os.path.getsize(EMBEDDINGS_FILE) > 0:
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.PicklingError) as e:
            print(f"Error loading embeddings.pkl: {e}. Regenerating embeddings.")
            os.remove(EMBEDDINGS_FILE)  # Remove corrupted file
    
    # If no valid embeddings.pkl, generate new embeddings
    embeddings = {}
    if not os.path.exists(PEOPLE_DIR):
        print(f"Error: {PEOPLE_DIR} directory not found.")
        return embeddings
    
    for filename in os.listdir(PEOPLE_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(PEOPLE_DIR, filename)
            try:
                embedding = DeepFace.represent(img_path, model_name='Facenet', enforce_detection=True)[0]["embedding"]
                embeddings[name] = np.array(embedding)
                print(f"Computed embedding for {name}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    if embeddings:
        try:
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Saved embeddings to {EMBEDDINGS_FILE}")
        except Exception as e:
            print(f"Error saving embeddings.pkl: {e}")
    else:
        print("No embeddings generated. Check if people/ directory contains valid images.")
    
    return embeddings

# Load embeddings at startup
embeddings = load_embeddings()
if not embeddings:
    print("Warning: No embeddings loaded. API will return 'Unknown' for all requests.")

@app.route('/search', methods=['POST'])
def search_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_path = 'temp.jpg'
    file.save(img_path)
    
    try:
        # Compute embedding for uploaded image
        query_embedding = DeepFace.represent(img_path, model_name='Facenet', enforce_detection=True)[0]["embedding"]
        query_embedding = np.array(query_embedding)
        
        # Find closest match
        min_distance = float('inf')
        matched_name = 'Unknown'
        
        if not embeddings:
            print("No embeddings available for comparison.")
            os.remove(img_path)
            return jsonify({'name': 'Unknown', 'distance': -1, 'error': 'No face embeddings available in database'}), 200
        
        for name, emb in embeddings.items():
            try:
                distance = cosine(query_embedding, emb)
                if np.isnan(distance) or np.isinf(distance):
                    print(f"Invalid distance for {name}: {distance}")
                    continue
                if distance < min_distance and distance < 0.4:  # Threshold for match
                    min_distance = distance
                    matched_name = name
            except Exception as e:
                print(f"Error comparing embedding for {name}: {e}")
                continue
        
        os.remove(img_path)
        # Convert Infinity to a finite value for JSON serialization
        response_distance = min_distance if np.isfinite(min_distance) else -1
        return jsonify({'name': matched_name, 'distance': response_distance})
    except Exception as e:
        if os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)