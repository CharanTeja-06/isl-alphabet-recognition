# Real-Time Indian Sign Language (ISL) Alphabet Recognition

A lightweight, real-time Indian Sign Language (ISL) static alphabet recognition system (A–Z) built using **MobileNetV2 transfer learning**, **TensorFlow/Keras**, and **OpenCV**.

The system captures video from a webcam, performs skin-color hand ROI detection, runs inference with temporal smoothing and hold-to-confirm logic, and displays predictions on an interactive HUD.

---

## 📌 Dataset

The project uses the **RealSign Indian Sign Language Dataset** containing static hand gesture images for letters A–Z across separate splits:

* **Training Set:** ~18,198 images
* **Validation Set:** ~2,579 images
* **Testing Set:** ~5,200 images

🔗 **Dataset Download:** [RealSign Indian Sign Language Dataset (.zip)](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset/blob/main/Dataset.zip)

Extract the dataset into the project root under the `dataset/` folder:
```text
dataset/
├── Training/    (A-Z)
├── Validation/  (A-Z)
└── Testing/     (A-Z)
```

---

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Model Architecture:** MobileNetV2 (Transfer Learning & Fine-Tuning)
* **Computer Vision:** OpenCV
* **Data Processing:** NumPy, `tf.data` API

---

## 🏗️ Project Architecture

```text
isl-alphabet-recognition/
├── dataset/
│   ├── Training/
│   ├── Validation/
│   └── Testing/
├── models/
│   └── isl_model.keras          # Trained model weights
├── src/
│   ├── data.py                  # tf.data pipeline, augmentation & preprocessing
│   ├── models.py                # MobileNetV2 architecture with custom classification head
│   ├── train.py                 # Two-phase training pipeline (frozen base + fine-tuning)
│   ├── evaluate.py              # Test set evaluation, per-letter accuracy & confusion matrix
│   └── webcam.py                # Real-time webcam inference with hand detection & smoothing
└── requirements.txt
```

---

## ⚙️ How It Works

1. **Preprocessing & Augmentation (`src/data.py`):**
   * Resizes images to `224x224` and applies native MobileNetV2 normalization (`[-1, 1]`).
   * Training includes random rotations, zoom, translation, horizontal flips, and brightness/contrast shifts to prevent overfitting.
2. **Model Design (`src/models.py`):**
   * MobileNetV2 base (pre-trained on ImageNet).
   * Classification Head: `GlobalAveragePooling2D` $\rightarrow$ `Dense(256, relu)` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Dense(26, softmax)`.
3. **Training Strategy (`src/train.py`):**
   * **Phase 1:** Feature extraction with frozen MobileNetV2 base (Adam, LR: `1e-3`).
   * **Phase 2:** Fine-tuning top MobileNetV2 layers from layer 100 onwards (Adam, LR: `1e-4`).
4. **Webcam Inference (`src/webcam.py`):**
   * **Hand Detection:** YCrCb skin-color segmentation detects the hand bounding box and eliminates false predictions when no hand is present.
   * **Confidence Filtering:** Predictions below the confidence threshold (`0.60`) are rejected as `Uncertain`.
   * **Temporal Smoothing:** Majority voting over a rolling 10-frame buffer.
   * **Hold-to-Confirm:** Requires holding a gesture for 15 frames to append the letter to a text buffer, complete with a visual progress bar.

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:
```bash
pip install tensorflow opencv-python numpy
```

---

### 2. Training the Model

To train the model on the dataset:
```bash
python src/train.py
```
The best checkpoint will be saved to `models/isl_model.keras`.

---

### 3. Evaluating Model Performance

Evaluate accuracy on the unseen test set, including per-letter breakdown and confusion matrix:
```bash
python src/evaluate.py
```

* **Test Accuracy:** ~91.5%

---

### 4. Running Real-Time Webcam Inference

Launch the live webcam recognition:
```bash
python src/webcam.py
```

**Webcam Controls:**
* **`Q`**: Quit application
* **`C`**: Clear text buffer
* **`B`**: Backspace (delete last confirmed letter)
