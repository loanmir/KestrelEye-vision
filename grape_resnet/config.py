import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Paths
MODEL_FOLDER = './model'
MODEL_SAVE_PATH = os.path.join(MODEL_FOLDER, 'resnet50_grape_model.h5')

# Create the folder automatically if it doesn't exist
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
    

TRAIN_DIR = '../data/classification_dataset/Final Training Data'
VAL_DIR = '../data/classification_dataset/Original Data/test'

# Configuration
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)

# Now unified because your folder names match!
CLASSES = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

def get_data_generators():

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )


    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes=CLASSES,
        shuffle=True
    )
    
    val_gen = ImageDataGenerator().flow_from_directory(
        VAL_DIR,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes=CLASSES, # Uses the exact same list
        shuffle=False
    )
    return train_gen, val_gen

def build_resnet50():
    base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAvgPool2D(),
        tf.keras.layers.Dropout(0.3),
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model