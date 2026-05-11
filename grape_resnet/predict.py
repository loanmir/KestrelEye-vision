import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from config import MODEL_SAVE_PATH, CLASSES, TARGET_SIZE

def predict_single_image(img_path):
    # 1. Verification
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        return
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at '{MODEL_SAVE_PATH}'. Please run train.py first.")
        return

    # 2. Load the Model
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # 3. Preprocess the Image (The most critical step)
    # Load the image and resize it to 224x224 as required by ResNet50
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    
    # Expand dimensions: (224, 224, 3) -> (1, 224, 224, 3)
    # ResNet50 expects a 'batch' of images, even if it's just one
    img_array = np.expand_dims(img_array, axis=0) 
    
    # 4. Make Prediction
    print(f"Classifying: {img_path}...")
    predictions = model.predict(img_array)
    
    # 5. Extract Results
    score = predictions[0]
    predicted_class_idx = np.argmax(score)
    confidence = score[predicted_class_idx] * 100
    
    print("\n" + "="*30)
    print(f"RESULT: {CLASSES[predicted_class_idx]}")
    print(f"CONFIDENCE: {confidence:.2f}%")
    print("="*30)

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Predict grape disease for a single leaf image.")
    parser.add_argument("path", type=str, help="Path to the image file (e.g., ./leaf.jpg)")
    
    args = parser.parse_args()
    predict_single_image(args.path)