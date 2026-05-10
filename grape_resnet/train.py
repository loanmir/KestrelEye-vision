from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import get_data_generators, build_resnet50, MODEL_SAVE_PATH

def run_training():
    train_gen, val_gen = get_data_generators()
    model = build_resnet50()
    
    # Callbacks to handle training efficiency
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(
            MODEL_SAVE_PATH, 
            save_best_only=True, 
            monitor='val_accuracy', 
            mode='max'
        )
    ]
    
    print("Starting ResNet50 Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=callbacks
    )
    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")
    return history

if __name__ == "__main__":
    run_training()