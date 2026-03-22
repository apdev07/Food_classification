# 🚀 Quick Start Guide

Get your Food Image Classifier running in 5 minutes!

---

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2: Prepare Dataset

### Option A: Use Your Own Images

Create this folder structure:
```
dataset/
├── pizza/          (put pizza images here)
├── burger/         (put burger images here)
├── dosa/           (put dosa images here)
├── biryani/        (put biryani images here)
└── salad/          (put salad images here)
```

**Recommended:** 50-100 images per class for good accuracy.

### Option B: Download Sample Images

```bash
python download_sample_data.py
```

This downloads a few sample images for testing.

---

## Step 3: Train the Model

```bash
python train_model.py
```

Training takes 5-15 minutes on CPU, faster on GPU.

**What happens:**
- ✅ Images are loaded and preprocessed
- ✅ MobileNetV2 model is loaded (frozen layers)
- ✅ Custom layers are trained for 10 epochs
- ✅ Best model is saved automatically
- ✅ Training graphs are generated

---

## Step 4: Run the Web App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

Upload a food image and get instant predictions!

---

## Alternative: Command Line Prediction

```bash
python predict.py dataset/pizza/pizza1.jpg
```

---

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `train_model.py` | Main training script |
| `app.py` | Streamlit web application |
| `predict.py` | Command-line prediction |
| `train_with_finetuning.py` | Advanced training with fine-tuning |
| `download_sample_data.py` | Download sample images |
| `Food_Classification_Notebook.ipynb` | Jupyter notebook version |

---

## 🎯 Expected Output

After training, you'll have:

```
food_classification_project/
├── food_classifier_model.h5          # Best model (saved during training)
├── food_classifier_final.keras       # Final model
├── class_indices.json                # Class name mapping
├── training_history.png              # Accuracy/loss graphs
└── sample_predictions.png            # Sample prediction visualization
```

---

## 💡 Tips for Better Accuracy

1. **More Data = Better Results**
   - Aim for 100+ images per class
   - Variety in angles, lighting, backgrounds

2. **Check Your Images**
   - Make sure images are in correct folders
   - Remove corrupted or irrelevant images

3. **Increase Epochs**
   - Edit `EPOCHS = 15` in train_model.py
   - But watch for overfitting!

4. **Try Fine-Tuning**
   - Use `train_with_finetuning.py` for potentially better results
   - Unfreezes top layers of MobileNetV2

---

## 🐛 Common Issues

### "Dataset directory not found"
Create the `dataset` folder with subfolders as shown above.

### "Out of memory"
Reduce `BATCH_SIZE` to 16 or 8 in train_model.py

### "Low accuracy"
- Add more training images
- Check if images are labeled correctly
- Train for more epochs

### "Module not found"
```bash
pip install -r requirements.txt
```

---

## 📊 Understanding Results

### Training Output Example:
```
Epoch 1/10
10/10 [==============================] - 15s 1s/step - loss: 1.2345 - accuracy: 0.5234 - val_loss: 0.9876 - val_accuracy: 0.6543
```

**What to look for:**
- `accuracy`: Training accuracy (should increase)
- `val_accuracy`: Validation accuracy (should be close to training)
- If `val_accuracy` << `accuracy`: Overfitting!

### Good Results:
- Training Accuracy: 85-95%
- Validation Accuracy: 75-90%
- Gap between them: < 10%

---

## 🎓 Learning Resources

- **MobileNetV2 Paper**: https://arxiv.org/abs/1801.04381
- **Transfer Learning Guide**: https://www.tensorflow.org/tutorials/images/transfer_learning
- **Keras Docs**: https://keras.io/

---

Happy Classifying! 🍕🍔🍘🍛🥗
