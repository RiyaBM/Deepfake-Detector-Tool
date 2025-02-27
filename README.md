# **Deepfake Detection Tool**  

## **Table of Contents**  
- [Introduction](#introduction)  
- [Features](#features)  
- [Dataset Structure](#dataset-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Training the Model](#training-the-model)  
  - [Making Predictions](#making-predictions)  
- [Requirements](#requirements)  

---

## **Introduction**  

This project is designed to detect deepfake images using an **ensemble model** that combines **EfficientNetB3** and **Xception** architectures. By leveraging transfer learning and fine-tuning, the model improves accuracy in distinguishing real and fake images.  

## **Overview**
The project consists of two scripts:  
**`ensemble.py`**
- Loads and preprocesses images from designated directories.
- Constructs an ensemble model by merging EfficientNetB3 and Xception.
- Trains the ensemble in two phases:
  - Initial Training: With frozen base layers.
  - Fine-Tuning: Unfreezes deeper layers for enhanced performance.
- Saves the final trained model as deepfake_image_model.h5.

**`predict.py`**
- Loads the saved model.
- Provides a prediction function to classify new images as "real" or "fake" by processing each image and passing it through the ensemble model. 

---

## **Features**  

✅ **Ensemble Learning:** Combines two powerful CNN models for better accuracy.  
✅ **Transfer Learning:** Utilizes pre-trained weights from ImageNet.  
✅ **Two-Phase Training:** Initial training with frozen layers, followed by fine-tuning.  
✅ **TensorFlow Data Pipelines:** Efficient image loading, preprocessing, and augmentation.  
✅ **GPU Support:** Automatically detects and uses available GPUs.  
✅ **Easy Deployment:** Pre-trained model can be used for quick predictions.  

---

## **Dataset Structure**  

Organize your dataset into the following structure:  

```
Dataset/
├── Train/
│   ├── real/
│   └── fake/
├── Validation/
│   ├── real/
│   └── fake/
└── Test/
    ├── real/
    └── fake/
Cls_pic/  # Additional images to improve training
```

Ensure that images are in `.jpg` or `.png` format. Update the paths in `ensemble.py` accordingly.  

---

## **Installation**  

### **1. Clone the repository**  

```bash
git clone https://github.com/yourusername/deepfake-image-detection.git
cd deepfake-image-detection
```

### **2. Create a virtual environment (Optional but recommended)**  

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install dependencies**  

```bash
pip install tensorflow numpy
```

---

## **Usage**  

### **Training the Model**  

To train the ensemble model, run:  

```bash
python ensemble.py
```

This script will:  
- Load and preprocess images  
- Train the ensemble model with frozen layers  
- Fine-tune deeper layers  
- Save the trained model as `deepfake_image_model.h5`  

---

### **Making Predictions**  

Once the model is trained, use `predict.py` to classify new images:  

```bash
python predict.py
```

The script will:  
- Load the saved model  
- Process and normalize the input image  
- Output whether the image is "real" or "fake"  

---

## **Requirements**  

- Python 3.6+  
- TensorFlow 2.x  
- NumPy  
- GPU (Optional but recommended for faster training)  

---
