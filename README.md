# Face Recognition API

A simple Flask-based REST API for face recognition using DeepFace and Facenet. The API computes facial embeddings for a dataset of images and allows searching for the closest match to an uploaded face image.

## Features

- Computes and stores facial embeddings from a dataset.
- REST endpoint to search for the closest face match.
- Uses DeepFace (Facenet model) for embedding extraction.
- Supports CORS for frontend integration (e.g., React).
- Handles corrupted or missing embeddings gracefully.

## Project Structure

```
face-recognition/
├── api/
│   └── app.py           # Flask API source code
├── dataset/             # Folder containing known face images (.jpg, .jpeg, .png)
├── embeddings.pkl       # Pickled facial embeddings (auto-generated)
└── README.md
```

## Requirements

- Python 3.7+
- Flask
- deepface
- numpy
- opencv-python
- scipy
- flask-cors
- tf-keras

Install dependencies with:

```bash
pip install flask deepface numpy opencv-python scipy flask-cors tf-keras
```

## Usage

1. **Prepare Dataset**

   Place images of known people in the `dataset/` directory. Each image filename (without extension) will be used as the person's name.

2. **Run the API**

   ```bash
   cd api
   python app.py
   ```

   The API will compute embeddings for all images in `../dataset/` and save them to `embeddings.pkl`.

3. **Search for a Face**

   Send a POST request to `/search` with an image file:

   ```bash
   curl -X POST -F "image=@/path/to/face.jpg" http://localhost:5000/search
   ```

   **Response:**
   ```json
   {
     "name": "person_name",
     "distance": 0.32
   }
   ```

   If no match is found, `"name"` will be `"Unknown"`.

## API Endpoints

- `POST /search`
  - Form-data: `image` (file)
  - Returns: JSON with `name` and `distance`

## Notes

- The matching threshold is set to 0.4 (cosine distance).
- If no embeddings are available, the API will return `"Unknown"` for all requests.
- Embeddings are cached in `embeddings.pkl` for faster startup.

## License

MIT License