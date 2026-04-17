import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from ultralytics import YOLO
print("TF VERSION:", tf.__version__)
print("TF KERAS ENABLED")

print("Loading yolo...")
yolo = YOLO("yolov8n.pt")

print("Loading emotion model...")
try:
    emo = tf.keras.models.load_model("models/emotion_model.h5", compile=False)
    print("Emotion model OK")
except Exception as e:
    print("Emotion error:", e)

print("Loading breed model...")
try:
    breed = tf.keras.models.load_model("models/breed_model.h5", compile=False)
    print("Breed model OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Breed error:", e)
