import os
if "TF_USE_LEGACY_KERAS" in os.environ:
    del os.environ["TF_USE_LEGACY_KERAS"]
import tensorflow as tf

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

try:
    print("Loading emotion weights...")
    emo = create_emotion_model()
    emo.load_weights("models/emotion_model.h5")
    print("Emotion success!")
    
    print("Loading breed weights...")
    breed = create_breed_model()
    breed.load_weights("models/breed_model.h5")
    print("Breed success!")
except Exception as e:
    import traceback
    traceback.print_exc()
