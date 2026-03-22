# 🍕 Food Image Classification System

A deep learning project that classifies food images into 5 categories using Transfer Learning with MobileNetV2.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)

---

## 📋 Project Overview

This project uses **Transfer Learning** with a pretrained **MobileNetV2** model to classify food images into 5 categories:
- 🍕 **Pizza**
- 🍔 **Burger**
- 🍘 **Dosa**
- 🍛 **Biryani**
- 🥗 **Salad**

### Key Features
- ✅ Transfer Learning with MobileNetV2 (pretrained on ImageNet)
- ✅ Data Augmentation (rotation, zoom, flip)
- ✅ 224x224 image preprocessing
- ✅ Training/Validation split (80/20)
- ✅ Interactive Streamlit web app
- ✅ Sample predictions visualization

---

## 🗂️ Project Structure

```
food_classification_project/
│
├── train_model.py          # Main training script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── dataset/               # Create this folder with your images
│   ├── pizza/
│   ├── burger/
│   ├── dosa/
│   ├── biryani/
│   └── salad/
│
└── (generated after training)
    ├── food_classifier_model.h5      # Best model
    ├── food_classifier_final.keras   # Final model
    ├── class_indices.json            # Class mapping
    ├── training_history.png          # Training graphs
    └── sample_predictions.png        # Sample predictions
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset

Create a `dataset` folder with subfolders for each food class:

```bash
mkdir dataset
cd dataset
mkdir pizza burger dosa biryani salad
```

Add images to each folder (at least 50-100 images per class for good results).

**Where to get images:**
- [Kaggle Food-101 Dataset](https://www.kaggle.com/datasets/kmader/food41)
- [Google Images](https://images.google.com) (use browser extensions to download)
- Take your own photos!

### Step 3: Train the Model

```bash
python train_model.py
```

This will:
1. Load and preprocess images
2. Build the MobileNetV2-based model
3. Train for 10 epochs
4. Save the best model
5. Generate training plots

### Step 4: Run the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Model Architecture

```
Input (224x224x3)
    ↓
MobileNetV2 (Frozen, pretrained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense (128 units, ReLU)
    ↓
BatchNormalization
    ↓
Dropout (0.5)
    ↓
Dense (5 units, Softmax)
    ↓
Output (Class probabilities)
```

**Total Parameters:** ~3.5 Million (mostly frozen in base model)

**Trainable Parameters:** ~130,000

---

## 🔧 Configuration

You can modify these parameters in `train_model.py`:

```python
# Model parameters
IMG_SIZE = 224          # Input image size
BATCH_SIZE = 32         # Training batch size
EPOCHS = 10             # Number of training epochs
LEARNING_RATE = 0.0001  # Adam optimizer learning rate

# Classes to classify
CLASS_NAMES = ['pizza', 'burger', 'dosa', 'biryani', 'salad']
```

---

## 📈 Expected Results

With a good dataset (100+ images per class), you should achieve:

| Metric | Expected Value |
|--------|---------------|
| Training Accuracy | 85-95% |
| Validation Accuracy | 75-90% |
| Training Time | 5-15 minutes (CPU) |
| Model Size | ~15 MB |

---

## 🛠️ Troubleshooting

### Issue: "Dataset directory not found"
**Solution:** Create the `dataset` folder with proper structure as shown above.

### Issue: "Out of memory"
**Solution:** Reduce `BATCH_SIZE` to 16 or 8 in `train_model.py`.

### Issue: Low accuracy
**Solutions:**
- Add more training images (at least 50 per class)
- Increase `EPOCHS` to 15-20
- Check if images are labeled correctly
- Try unfreezing some top layers of MobileNetV2

### Issue: "Module not found"
**Solution:** Install all requirements:
```bash
pip install -r requirements.txt
```

---

## 📝 Code Explanation

### 1. Data Preprocessing
```python
# Training data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize to [0,1]
    rotation_range=30,         # Random rotation
    zoom_range=0.2,            # Random zoom
    horizontal_flip=True,      # Random flip
    validation_split=0.2       # 80/20 split
)
```

### 2. Model Building
```python
# Load pretrained MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,      # Remove final layer
    weights='imagenet'      # Use pretrained weights
)
base_model.trainable = False  # Freeze base layers
```

### 3. Training
```python
model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stopping]
)
```

---

## 🎯 Future Improvements

- [ ] Add more food classes
- [ ] Fine-tune top layers of MobileNetV2
- [ ] Implement Grad-CAM for visualization
- [ ] Deploy to cloud (Heroku/AWS)
- [ ] Add confidence threshold filtering
- [ ] Collect user feedback for model improvement

---

## 📚 References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras Documentation](https://keras.io/)

---

## 👨‍💻 Author

**BTech AIML Student Project**

Feel free to modify and extend this project for your learning!

---

## 📄 License

This project is for educational purposes. Feel free to use and modify.

---

## 🙏 Acknowledgments

- TensorFlow Team for the amazing framework
- Streamlit for the easy-to-use web app framework
- MobileNetV2 authors for the efficient architecture
