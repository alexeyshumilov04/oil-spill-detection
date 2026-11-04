import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.config import MODELS_DIR, DATA_DIR
from src.models.losses import dice_loss

def predict_image(image_path):
    model_path = os.path.join(MODELS_DIR, "my_model.h5")
    model = load_model(model_path, custom_objects={'dice_loss': dice_loss})

    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_label = "Oil Spill" if class_idx == 1 else "No Spill"
    
    print(f"Prediction: {class_label} (Confidence: {prediction[0][class_idx]:.2f})")
    return class_label, prediction[0][class_idx]