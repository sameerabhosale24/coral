🪸 Coral Bleaching Detector: Deep Learning for Marine Conservation
Coral Bleaching Detector is a specialized computer vision application designed to automate the classification of coral health using deep learning. By utilizing a fine-tuned ResNet-18 model, the system identifies whether a coral specimen is "Healthy" or "Bleached" based on visual pigment density and skeletal textures. This tool aims to assist researchers by providing objective, high-speed diagnostics for underwater imagery.

🚀 Key Features
Custom CNN Architecture: Built on the ResNet-18 backbone, optimized for binary classification.
Advanced Training Pipeline: Implements data augmentation (random horizontal flips, 15° rotations, and color jitter) to handle the varied lighting and angles of underwater photography.
Mixed-Precision Training: Utilizes torch.cuda.amp for significantly faster training and lower memory consumption on modern GPUs.
Interactive Diagnostic Interface: A dedicated Streamlit frontend (as seen in image_25e0ab.jpg) that allows users to upload images and receive instant health reports with confidence scores.
Standardized Preprocessing: Uses ImageNet-derived Z-score normalization to ensure consistent feature extraction across diverse test sets.

🛠️ Technical Stack
Deep Learning Framework: PyTorch.
Web UI: Streamlit.
Data Science Tools: Scikit-learn (Classification Reports), Seaborn/Matplotlib (Confusion Matrices).
Core Model: ResNet-18.

## ⚙️ Setup

Install dependencies:
pip install -r requirements.txt

---

## 📂 Dataset

Dataset is not included due to size limitations.

Download from:
<PASTE https://drive.google.com/drive/folders/1atR8o5x5MPUL1qDLMTk3zPD1LKiV8sRJ?usp=drive_link >

Place it inside:
data/

---

## 🤖 Model

Download pretrained model from:
<PASTE https://drive.google.com/file/d/1AgJBadb4q3kTxSDK_QcuxyG4XGKgYOlP/view?usp=drive_link>

Place it in root folder as:
best_resnet18_coral.pth

---

## 🚀 Usage

Train:
python train_coral.py

Test:
python test_coral.py

Evaluate:
python evaluate.py

Web UI:
python app.py
---

## ⚠️ Note
Dataset and model are hosted externally due to GitHub size limits.
