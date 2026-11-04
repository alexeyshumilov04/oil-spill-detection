import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.config import DATA_DIR, MODELS_DIR
from src.models.cnn_model import build_cnn_model
from src.models.losses import dice_loss

def train_model(epochs=5, batch_size=32):
    """
    Загружает данные из data/processed, обучает CNN и сохраняет модель.
    """
    base_path = os.path.join(DATA_DIR, "processed")
    oil_dir = os.path.join(base_path, "oil_spill")
    no_dir = os.path.join(base_path, "no_spill")
    
    if not os.path.exists(oil_dir) or not os.path.exists(no_dir):
        raise FileNotFoundError("Dataset not found! Run scripts/download_data.py first.")

    print("Loading dataset...")
    oil_paths = [(os.path.join(oil_dir, f), 1) for f in os.listdir(oil_dir)]
    no_paths = [(os.path.join(no_dir, f), 0) for f in os.listdir(no_dir)]
    data = oil_paths + no_paths
    
    X, y = [], []
    for path, label in data:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64)) / 255.0
        X.append(img)
        y.append(label)

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=2)
    
    print(f"Dataset loaded: {X.shape[0]} images.")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Building model...")
    model = build_cnn_model()
    model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])
    
    print("Starting training...")
    model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test)
    )
    
    # Сохранение модели
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "my_model.h5")
    model.save(model_path)
    
    print(f"✅ Model successfully saved to {model_path}")
    return model_path