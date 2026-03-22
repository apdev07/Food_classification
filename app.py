"""
Food Image Classification - Streamlit App
=========================================
A simple web app to classify food images using the trained model.

Run with: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Page configuration
st.set_page_config(
    page_title="Food Classifier",
    page_icon="🍕",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        text-align: center;
        color: #FF6B6B;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL AND CLASS INDICES
# =============================================================================

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)."""
    
    # Try different model file formats
    model_files = ['food_classifier_final.keras', 'food_classifier_model.h5']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = tf.keras.models.load_model(model_file)
                return model
            except Exception as e:
                st.error(f"Error loading {model_file}: {e}")
                continue
    
    return None

@st.cache_data
def load_class_indices():
    """Load class indices mapping."""
    
    if os.path.exists('class_indices.json'):
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        return class_indices
    
    # Default classes if file not found
    return {
        "0": "pizza",
        "1": "burger", 
        "2": "dosa",
        "3": "biryani",
        "4": "salad"
    }

def preprocess_image(image):
    """
    Preprocess the uploaded image for prediction.
    
    Steps:
    1. Resize to 224x224
    2. Convert to array
    3. Normalize pixel values
    4. Add batch dimension
    """
    # Resize image
    image = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Ensure image has 3 channels (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(image, model, class_indices):
    """
    Make prediction on the preprocessed image.
    """
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)
    
    # Get predicted class and confidence
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    
    # Get class name
    class_name = class_indices.get(str(predicted_idx), f"Class {predicted_idx}")
    
    # Get all predictions for displaying probabilities
    all_predictions = {
        class_indices.get(str(i), f"Class {i}"): float(predictions[0][i]) * 100
        for i in range(len(predictions[0]))
    }
    
    # Sort by probability
    all_predictions = dict(sorted(all_predictions.items(), 
                                  key=lambda x: x[1], reverse=True))
    
    return class_name, confidence, all_predictions

# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    # Title and description
    st.markdown("<h1 class='title'>🍕 Food Image Classifier</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; font-size: 18px; color: #666;'>
            Upload a food image and let AI identify it!
        </p>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("""
            ⚠️ Model not found! Please train the model first by running:
            ```
            python train_model.py
            ```
        """)
        return
    
    # Load class indices
    class_indices = load_class_indices()
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
            This app uses a deep learning model (MobileNetV2) 
            trained to classify 5 types of food:
            - 🍕 Pizza
            - 🍔 Burger  
            - 🍘 Dosa
            - 🍛 Biryani
            - 🥗 Salad
        """)
        
        st.header("📊 Model Info")
        st.write(f"Classes: {len(class_indices)}")
        st.write(f"Input Size: 224×224")
        st.write(f"Base Model: MobileNetV2")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a food image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of pizza, burger, dosa, biryani, or salad"
    )
    
    # Camera input option
    st.markdown("<p style='text-align: center; color: #888;'>OR</p>", unsafe_allow_html=True)
    camera_image = st.camera_input("Take a photo")
    
    # Use camera image if uploaded file is None
    if camera_image is not None and uploaded_file is None:
        uploaded_file = camera_image
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🔮 Prediction")
            
            # Make prediction
            with st.spinner('Analyzing image...'):
                predicted_class, confidence, all_preds = predict(image, model, class_indices)
            
            # Display result
            st.markdown(f"""
                <div class='prediction-box'>
                    <h2 style='color: #FF6B6B; margin: 0;'>{predicted_class.upper()}</h2>
                    <p style='font-size: 24px; margin: 10px 0;'>
                        Confidence: <b>{confidence:.2f}%</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence indicator
            if confidence >= 80:
                st.success("✅ High confidence prediction!")
            elif confidence >= 50:
                st.warning("⚠️ Moderate confidence")
            else:
                st.error("❌ Low confidence - try another image")
        
        # Show all probabilities
        st.subheader("📊 All Class Probabilities")
        
        # Create progress bars for each class
        for class_name, prob in all_preds.items():
            col_label, col_bar = st.columns([1, 3])
            with col_label:
                st.write(f"**{class_name.capitalize()}**")
            with col_bar:
                st.progress(prob / 100)
                st.caption(f"{prob:.2f}%")
        
        # Additional info
        st.divider()
        st.caption("""
            💡 **Tip**: For best results, upload clear images with the food item 
            centered and well-lit. The model works best with single food items.
        """)

if __name__ == "__main__":
    main()
