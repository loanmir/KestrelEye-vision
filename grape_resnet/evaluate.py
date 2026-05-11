import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import get_data_generators, MODEL_SAVE_PATH, CLASSES

def run_detailed_evaluation():
    # 1. Load Data and Model
    _, test_gen = get_data_generators() # Use the validation/test generator
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at {MODEL_SAVE_PATH}. Please train first.")
        return

    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print("Model loaded successfully. Starting evaluation...")

    # 2. Statistical Evaluation (Optimization from .ipynb)
    # Reset generator to ensure we start from the first image
    test_gen.reset()
    
    # Predict on the entire dataset
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    filenames = test_gen.filenames

    # 3. Per-Image Results
    print("\n" + "="*50)
    print(f"{'IMAGE FILENAME':<40} | {'PREDICTION':<15} | {'ACTUAL':<15} | {'STATUS'}")
    print("-" * 85)
    
    correct_count = 0
    for idx in range(len(filenames)):
        pred_label = CLASSES[predicted_classes[idx]]
        true_label = CLASSES[true_classes[idx]]
        status = "Good" if pred_label == true_label else "Bad"
        if status == "Good": correct_count += 1
        
        # Prints every image result as requested
        print(f"{filenames[idx]:<40} | {pred_label:<15} | {true_label:<15} | {status}")

    # 4. Summary Metrics 
    print("\n" + "="*50)
    print("DETAILED PERFORMANCE REPORT")
    print("="*50)
    
    # Print the Classification Report (Precision, Recall, F1-Score)
    print(classification_report(true_classes, predicted_classes, target_names=CLASSES))

    # 5. Confusion Matrix Visualization
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix: Grape Disease Classification')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    run_detailed_evaluation()