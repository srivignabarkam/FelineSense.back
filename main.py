from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import shutil

app = FastAPI(title="Cat Emotion AI API")

# Allow Next.js frontend to access FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
UPLOAD_DIR = os.path.join(BASE_PATH, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

emotion_model_path = os.path.join(MODELS_DIR, "emotion_model.h5")
breed_model_path = os.path.join(MODELS_DIR, "breed_model.h5")

def create_emotion_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

def create_breed_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(12, activation='softmax')
    ])

# Initialize models
emotion_model = None
breed_model = None
yolo_model = None

try:
    if os.path.exists(emotion_model_path):
        emotion_model = create_emotion_model()
        emotion_model.load_weights(emotion_model_path)
except Exception as e:
    print("Warning: Failed to load emotion model:", e)

try:
    if os.path.exists(breed_model_path):
        breed_model = create_breed_model()
        breed_model.load_weights(breed_model_path)
except Exception as e:
    print("Warning: Failed to load breed model:", e)

try:
    yolo_model = YOLO("yolov8n.pt") 
except Exception as e:
    print("Warning: Failed to load YOLO model:", e)

emotion_classes = ["Angry", "Happy", "Sad"]
breed_classes = [
    "Abyssinian","Bengal","Birman","Bombay","British_Shorthair",
    "Egyptian_Mau","Maine_Coon","Persian","Ragdoll",
    "Russian_Blue","Siamese","Sphynx"
]

emoji_map = {
    "Happy": "😺",
    "Angry": "😾",
    "Sad": "😿"
}

@app.get("/")
def home():
    return {"status": "ok", "message": "Cat Emotion AI API is running"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    img = cv2.imread(filepath)
    if img is None:
        if os.path.exists(filepath): os.remove(filepath)
        return {"status": "error", "message": "Invalid Image"}
    
    if yolo_model is None or emotion_model is None or breed_model is None:
        if os.path.exists(filepath): os.remove(filepath)
        return {
            "status": "error", 
            "message": "Models not found or failed to load. Please place .h5 models in backend/models folder."
        }

    results = yolo_model(filepath)
    
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 15: # class 15 is cat
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cat_img = r.orig_img[y1:y2, x1:x2]
                
                resized = cv2.resize(cat_img, (128,128))
                normalized = resized / 255.0
                input_img = np.expand_dims(normalized, axis=0)
                
                emotion_pred = emotion_model.predict(input_img)
                emotion = emotion_classes[np.argmax(emotion_pred)]
                
                breed_pred = breed_model.predict(input_img)
                breed = breed_classes[np.argmax(breed_pred)]
                
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
                return {
                    "status": "success",
                    "breed": breed,
                    "emotion": emotion,
                    "emoji": emoji_map.get(emotion, "")
                }
                
    if os.path.exists(filepath):
        os.remove(filepath)
        
    return {"status": "error", "message": "No Cat Detected"}
