# 🛰️ Oil Spill Detection using Sentinel-2 & CNNs

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-red.svg)](https://spark.apache.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)

A scalable, end-to-end Machine Learning pipeline for automated **oil spill detection** using Sentinel-2 satellite imagery, Convolutional Neural Networks (CNNs), and Apache Spark for distributed data processing. 

Designed to monitor maritime ecosystems, including regions like the Gulf of Mexico, Mumbai Coast, and Port of Los Angeles.

---

## Example Detection Output

Below is an example of oil spill detection on a Sentinel-2 image from the **Indonesia Sea** region:

| Annotated Detection | Detected Mask |
|----------------|----------------|
| ![Detection](datasets\processed\Indonesia_Sea\detection.png) | ![Mask](datasets\processed\Indonesia_Sea\mask.png) |

---

## ✨ Key Features

- **Automated Data Acquisition:** Fetches high-resolution Sentinel-2 imagery directly via Sentinel Hub API.
- **Distributed Processing:** Utilizes Apache Spark to parallelize image downloading, preprocessing, and dataset generation across multiple geographic locations.
- **Deep Learning:** Custom CNN architecture (TensorFlow/Keras) trained to classify image patches as `oil_spill` or `no_spill`.
- **Computer Vision Preprocessing:** Employs OpenCV for thresholding, contour detection, and patch tiling.
- **Modular Architecture:** Clean separation of concerns (data ingestion, model training, inference).
- **Dockerized:** Ready for deployment in containerized environments.

---

## 📂 Project Structure

```text
OilSpillDetection/
│
├── datasets/                  # Data directory 
│   ├── raw/                   # Raw downloaded images (oil_spill / no_spill)
│   ├── processed/             # Tiled patches ready for training
│   └── test_images/           # Holdout images for validation
│
├── model_weights/             # Saved Keras models (.h5 files)
│   └── my_model.h5
│
├── scripts/                   # Entry points for execution
│   ├── download_data.py       # Fetches & processes satellite imagery
│   ├── train_model.py         # Trains the CNN model
│   └── predict.py             # Runs inference on a test image
│
├── src/                       # Core source code
│   ├── configs/               # Configuration files
│   ├── data/                  # Sentinel API & Dataset builder logic
│   ├── inference/             # Prediction pipelines
│   ├── models/                # CNN architecture & loss functions
│   └── config.py              # Environment variable loader
│
├── .env.example               # Example environment variables
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Project dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- Java 8+ (Required for Apache Spark)
- A [Sentinel Hub](https://www.sentinel-hub.com/) account to obtain API credentials.

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/alexeyshumilov04/OilSpillDetection.git
cd OilSpillDetection
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Environment Variables
Rename `.env.example` to `.env` and add your Sentinel Hub credentials:

```env
SH_INSTANCE_ID=your_instance_id
SH_CLIENT_ID=your_client_id
SH_CLIENT_SECRET=your_client_secret
```

---

## 🛠️ Usage

The project is divided into three main executable scripts:

### Step 1: Download & Process Data
Fetches imagery for target locations using Apache Spark and generates tiled patches (64x64) for training.
```bash
python scripts/download_data.py
```

### Step 2: Train the CNN Model
Loads the generated dataset, trains the CNN with a custom Dice Loss function, and saves the model to `model_weights/my_model.h5`.
```bash
python scripts/train_model.py
```

### Step 3: Run Inference
Predicts whether a given test image contains an oil spill.
```bash
python scripts/predict.py
```

---

## 🐳 Docker Support

To build and run the project in an isolated container:

```bash
docker build -t oil-spill-detection .
docker run --env-file .env oil-spill-detection
```

---

## 📊 Results Summary

*   **CNN Accuracy:** 94.5%
*   **True Positive Rate:** 95.0%
*   **True Negative Rate:** 96.5%
*   **Parallel Speedup (8 cores):** 3.85×

---

## 📝 Workflow Overview

1.  **Acquisition:** Sentinel-2 L1C imagery is requested via API.
2.  **Tiling:** Images are split into 64x64 patches.
3.  **Labeling:** OpenCV thresholding is applied to determine if a patch contains oil (threshold > 10%).
4.  **Training:** Patches are fed into the CNN (`src/models/cnn_model.py`).
5.  **Inference:** The trained model predicts spill presence on unseen data.