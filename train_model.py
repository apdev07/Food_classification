"""
Food Image Classification System
================================
A deep learning project using Transfer Learning with MobileNetV2
to classify food items: Pizza, Burger, Dosa, Biryani, and Salad.

Author: BTech AIML Student Project
"""

# =============================================================================
# SECTION 1: IMPORT LIBRARIES
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("Food Image Classification System")
print("Using Transfer Learning with MobileNetV2")
print("=" * 60)

# =============================================================================
# SECTION 2: CONFIGURATION & HYPERPARAMETERS
# =============================================================================

# Dataset paths - update these according to your folder structure
DATA_DIR = "dataset"  # Main folder containing class subfolders

# Model parameters
IMG_SIZE = 224          # MobileNetV2 expects 224x224 images
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

# Class names (folders should be named exactly like this)
CLASS_NAMES = ['pizza', 'burger', 'dosa', 'biryani', 'salad']
NUM_CLASSES = len(CLASS_NAMES)

print(f"\nConfiguration:")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Classes: {CLASS_NAMES}")

# =============================================================================
# SECTION 3: DATA PREPROCESSING & AUGMENTATION
# =============================================================================

def create_data_generators():
    """
    Create training and validation data generators with augmentation.
    
    Training data: Augmented (rotation, zoom, flip) to improve generalization
    Validation data: Only resized and normalized (no augmentation)
    """
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalize pixel values to [0,1]
        rotation_range=30,            # Random rotation up to 30 degrees
        width_shift_range=0.2,        # Horizontal shift
        height_shift_range=0.2,       # Vertical shift
        zoom_range=0.2,               # Random zoom
        horizontal_flip=True,         # Random horizontal flip
        fill_mode='nearest',          # Fill strategy for new pixels
        validation_split=0.2          # 80% train, 20% validation
    )
    
    # Validation data generator (only rescaling, no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create generators from directory
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        classes=CLASS_NAMES,
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        classes=CLASS_NAMES,
        shuffle=False
    )
    
    return train_generator, validation_generator

# =============================================================================
# SECTION 4: BUILD THE MODEL
# =============================================================================

def build_model():
    """
    Build the food classification model using MobileNetV2 as base.
    
    Architecture:
    - MobileNetV2 (pretrained on ImageNet, frozen)
    - GlobalAveragePooling2D
    - Dense layer with ReLU activation
    - Dropout for regularization
    - Output Dense layer with Softmax
    """
    
    # Load pretrained MobileNetV2 without top layers
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,           # Exclude the final classification layer
        weights='imagenet'           # Use pretrained weights
    )
    
    # Freeze the base model layers (we don't want to train them)
    base_model.trainable = False
    
    print(f"\nBase model loaded: MobileNetV2")
    print(f"  Total layers in base model: {len(base_model.layers)}")
    print(f"  Trainable parameters: {base_model.count_params():,}")
    
    # Build the complete model
    model = keras.Sequential([
        # Pretrained base
        base_model,
        
        # Global average pooling (reduces dimensions)
        layers.GlobalAveragePooling2D(),
        
        # Dense hidden layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),          # Prevent overfitting
        
        # Output layer (5 classes)
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel compiled successfully!")
    model.summary()
    
    return model

# =============================================================================
# SECTION 5: TRAINING
# =============================================================================

def train_model(model, train_gen, val_gen):
    """
    Train the model with early stopping and checkpointing.
    """
    
    # Callbacks for better training
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            'food_classifier_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Stop early if validation accuracy stops improving
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    # Train the model
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# =============================================================================
# SECTION 6: EVALUATION & VISUALIZATION
# =============================================================================

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Loss plot
    axes[1].plot(history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    axes[1].plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\nTraining history plot saved as 'training_history.png'")
    plt.show()

def evaluate_model(model, val_gen):
    """
    Evaluate the model on validation set and print metrics.
    """
    
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    
    return val_loss, val_accuracy

# =============================================================================
# SECTION 7: PREDICTION ON SAMPLE IMAGES
# =============================================================================

def predict_sample_images(model, val_gen, num_samples=5):
    """
    Show predictions on sample validation images.
    """
    
    # Get a batch of validation images
    images, true_labels = next(val_gen)
    
    # Make predictions
    predictions = model.predict(images[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels[:num_samples], axis=1)
    
    # Plot results
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i])
        
        true_label = CLASS_NAMES[true_classes[i]]
        pred_label = CLASS_NAMES[predicted_classes[i]]
        confidence = predictions[i][predicted_classes[i]] * 100
        
        # Color code: green for correct, red for wrong
        color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
        
        ax.set_title(f"True: {true_label}\nPred: {pred_label}\n{confidence:.1f}%", 
                     color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    print("Sample predictions saved as 'sample_predictions.png'")
    plt.show()

# =============================================================================
# SECTION 8: SAVE CLASS INDICES
# =============================================================================

def save_class_indices(train_gen):
    """
    Save class indices mapping for later use in prediction.
    """
    import json
    
    # Invert the mapping (index -> class name)
    class_indices = {v: k for k, v in train_gen.class_indices.items()}
    
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)
    
    print(f"\nClass indices saved to 'class_indices.json'")
    print(f"Mapping: {class_indices}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to orchestrate the training pipeline.
    """
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"\nError: Dataset directory '{DATA_DIR}' not found!")
        print("Please create the following folder structure:")
        print("  dataset/")
        print("    ├── pizza/")
        print("    ├── burger/")
        print("    ├── dosa/")
        print("    ├── biryani/")
        print("    └── salad/")
        print("\nEach folder should contain images of that food item.")
        return
    
    # Step 1: Create data generators
    print("\nLoading and preprocessing data...")
    train_gen, val_gen = create_data_generators()
    
    # Step 2: Build the model
    model = build_model()
    
    # Step 3: Train the model
    history = train_model(model, train_gen, val_gen)
    
    # Step 4: Evaluate
    evaluate_model(model, val_gen)
    
    # Step 5: Plot training history
    plot_training_history(history)
    
    # Step 6: Show sample predictions
    predict_sample_images(model, val_gen)
    
    # Step 7: Save class indices
    save_class_indices(train_gen)
    
    # Step 8: Save final model in newer format
    model.save('food_classifier_final.keras')
    print("\nFinal model saved as 'food_classifier_final.keras'")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nFiles created:")
    print("  - food_classifier_model.h5 (best model)")
    print("  - food_classifier_final.keras (final model)")
    print("  - class_indices.json (class mapping)")
    print("  - training_history.png (training graphs)")
    print("  - sample_predictions.png (prediction samples)")

if __name__ == "__main__":
    main()
