import tensorflow as tf
import numpy as np
from config import get_data_generators, MODEL_SAVE_PATH, TRAIN_CLASSES

def evaluate_model():
    try:
        _, val_gen = get_data_generators()
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        
        print("\nEvaluating model on validation/test data...")
        loss, acc = model.evaluate(val_gen)
        print(f"Final Accuracy: {acc*100:.2f}%")
        
        # Example of getting a single prediction
        images, labels = next(val_gen)
        preds = model.predict(images)
        predicted_class = TRAIN_CLASSES[np.argmax(preds[0])]
        actual_class = TRAIN_CLASSES[int(labels[0])]
        
        print(f"\nSample Prediction:")
        print(f"Predicted: {predicted_class} | Actual: {actual_class}")
        
    except OSError:
        print(f"Error: Model file '{MODEL_SAVE_PATH}' not found. Please run train.py first.")

if __name__ == "__main__":
    evaluate_model()