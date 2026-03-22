"""
Standalone Prediction Script
============================
Make predictions on single images without the web app.

Usage:
    python predict.py path/to/image.jpg
"""

import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

def load_model_and_classes():
    """Load the trained model and class indices."""
    
    # Try to load model
    model_files = ['food_classifier_final.keras', 'food_classifier_model.h5']
    model = None
    
    for model_file in model_files:
        try:
            model = tf.keras.models.load_model(model_file)
            print(f"Loaded model: {model_file}")
            break
        except:
            continue
    
    if model is None:
        print("Error: No trained model found!")
        print("Please run 'python train_model.py' first.")
        sys.exit(1)
    
    # Load class indices
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
    except FileNotFoundError:
        print("Warning: class_indices.json not found, using default classes")
        class_indices = {str(i): name for i, name in enumerate(
            ['pizza', 'burger', 'dosa', 'biryani', 'salad'])}
    
    return model, class_indices

def preprocess_image(image_path):
    """Preprocess image for prediction."""
    
    # Load and resize image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(img)
    
    # Handle different image modes
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(image_path, model, class_indices):
    """Make prediction on a single image."""
    
    # Preprocess
    processed_img = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)
    
    # Get results
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    class_name = class_indices.get(str(predicted_idx), f"Class {predicted_idx}")
    
    return class_name, confidence, predictions[0]

def main():
    """Main function."""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py dataset/pizza/pizza1.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load model
    print("Loading model...")
    model, class_indices = load_model_and_classes()
    
    # Make prediction
    print(f"\nPredicting: {image_path}")
    print("-" * 40)
    
    try:
        class_name, confidence, all_probs = predict_image(image_path, model, class_indices)
        
        print(f"\n🍽️  Predicted: {class_name.upper()}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print("\nAll probabilities:")
        print("-" * 40)
        
        # Sort and display all probabilities
        for idx, prob in enumerate(all_probs):
            name = class_indices.get(str(idx), f"Class {idx}")
            bar = "█" * int(prob * 20)
            print(f"{name:10} | {bar:<20} | {prob*100:5.2f}%")
        
        # Confidence interpretation
        print("\n" + "-" * 40)
        if confidence >= 80:
            print("✅ High confidence prediction!")
        elif confidence >= 50:
            print("⚠️  Moderate confidence")
        else:
            print("❌ Low confidence - prediction may be unreliable")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
