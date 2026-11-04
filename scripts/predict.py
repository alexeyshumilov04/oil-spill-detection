import os
from src.inference.predict import predict_image
from src.config import DATA_DIR

def main():
    # Путь к тестовому изображению
    test_image_path = os.path.join(DATA_DIR, "test_images", "Venice_Lagoon_Test", "test_rgb.png")
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}. Please download it first.")
        return
        
    predict_image(test_image_path)

if __name__ == "__main__":
    main()