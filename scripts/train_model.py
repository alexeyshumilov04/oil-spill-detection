import sys
import os

# Добавляем корневую папку в sys.path, чтобы работали импорты из src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train import train_model

def main():
    try:
        # Здесь можно добавить argparse для передачи аргументов (например, epochs)
        train_model(epochs=5, batch_size=32)
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()