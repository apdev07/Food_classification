"""
Advanced Training with Fine-Tuning
==================================
This script includes fine-tuning of the top layers of MobileNetV2
for potentially better accuracy.

Training happens in two phases:
1. Phase 1: Train only the custom layers (frozen base)
2. Phase 2: Fine-tune top layers of MobileNetV2 along with custom layers
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
CLASS_NAMES = ['pizza', 'burger', 'dosa', 'biryani', 'salad']
NUM_CLASSES = len(CLASS_NAMES)

print("=" * 60)
print("Food Classification - Advanced Training with Fine-Tuning")
print("=" * 60)

def create_data_generators():
    """Create training and validation data generators."""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        classes=CLASS_NAMES
    )
    
    val_gen = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        classes=CLASS_NAMES
    )
    
    return train_gen, val_gen

def build_model():
    """Build model with MobileNetV2 base."""
    
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Initially freeze all layers
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model, base_model

def phase1_training(model, train_gen, val_gen):
    """
    Phase 1: Train only the custom layers with frozen base.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Training Custom Layers (Frozen Base)")
    print("=" * 60)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        ModelCheckpoint('phase1_best.h5', monitor='val_accuracy', 
                       save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    ]
    
    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    return history

def phase2_finetuning(model, base_model, train_gen, val_gen):
    """
    Phase 2: Fine-tune top layers of MobileNetV2.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-Tuning Top Layers")
    print("=" * 60)
    
    # Unfreeze the top 30 layers of MobileNetV2
    base_model.trainable = True
    
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Count trainable vs non-trainable
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    total_count = len(base_model.layers)
    print(f"Unfrozen layers: {trainable_count}/{total_count}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),  # Lower LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        ModelCheckpoint('food_classifier_finetuned.h5', 
                       monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    ]
    
    history = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    return history

def plot_combined_history(history1, history2):
    """Plot combined training history from both phases."""
    
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(epochs_range, acc, 'b-', label='Training', linewidth=2)
    axes[0].plot(epochs_range, val_acc, 'r-', label='Validation', linewidth=2)
    axes[0].axvline(x=len(history1.history['accuracy']), color='g', 
                    linestyle='--', label='Fine-tuning start')
    axes[0].set_title('Model Accuracy (Both Phases)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(epochs_range, loss, 'b-', label='Training', linewidth=2)
    axes[1].plot(epochs_range, val_loss, 'r-', label='Validation', linewidth=2)
    axes[1].axvline(x=len(history1.history['loss']), color='g', 
                    linestyle='--', label='Fine-tuning start')
    axes[1].set_title('Model Loss (Both Phases)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_finetuned.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline with fine-tuning."""
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found!")
        return
    
    # Create data generators
    print("Loading data...")
    train_gen, val_gen = create_data_generators()
    
    # Build model
    print("Building model...")
    model, base_model = build_model()
    model.summary()
    
    # Phase 1: Train custom layers
    history1 = phase1_training(model, train_gen, val_gen)
    
    # Phase 2: Fine-tune
    history2 = phase2_finetuning(model, base_model, train_gen, val_gen)
    
    # Plot combined history
    plot_combined_history(history1, history2)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Save model
    model.save('food_classifier_finetuned_final.keras')
    print("\nModel saved as 'food_classifier_finetuned_final.keras'")
    
    # Save class indices
    import json
    class_indices = {v: k for k, v in train_gen.class_indices.items()}
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)

if __name__ == "__main__":
    main()
