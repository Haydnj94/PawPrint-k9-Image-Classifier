import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import json

# Load the trained model
model = tf.keras.models.load_model("Models/final_dog_breed_model_MobileNet_Balanced.keras")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse the dictionary to get class labels
class_labels = {v: k for k, v in class_indices.items()}

def preprocess_image(img):
    target_size = (224, 224)
    img = img.convert("RGB")  # Ensure it's in RGB mode

    # Resize while maintaining aspect ratio
    img.thumbnail(target_size, Image.LANCZOS)

    # Create a blank (black) image with target size
    padded_img = Image.new("RGB", target_size, (0, 0, 0))  # Black padding (0,0,0)
    
    # Compute top-left position to paste resized image (center it)
    paste_x = (target_size[0] - img.size[0]) // 2
    paste_y = (target_size[1] - img.size[1]) // 2

    # Paste resized image onto padded background
    padded_img.paste(img, (paste_x, paste_y))

    # Convert to array and normalize
    img_array = image.img_to_array(padded_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array

# Streamlit UI
st.title("🐶 Dog Breed Classifier")
st.write("Upload an image of a dog, or capture one using your camera!")

# Live camera input (mobile-friendly)
camera_input = st.camera_input("Take a picture of the dog!")

if camera_input is not None:
    # Open the captured image
    img = Image.open(camera_input)
    st.image(img, caption="Captured Image", use_container_width=True)
    
    # Preprocess and predict
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    
    # Get the top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]  # Get indices of top 3 predictions (sorted in descending order)
    top_3_predictions = [(class_labels[idx], predictions[0][idx] * 100) for idx in top_3_indices]
    
    # Display the top 1 prediction
    predicted_class, confidence = top_3_predictions[0]
    
    # Show result
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Display the second and third most likely predictions
    st.subheader("Other likely breeds:")
    for breed, conf in top_3_predictions[1:]:
        st.write(f"{breed}: {conf:.2f}%")

# Allow uploading image (fallback option)
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess and predict
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    
    # Get the top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]  # Get indices of top 3 predictions (sorted in descending order)
    top_3_predictions = [(class_labels[idx], predictions[0][idx] * 100) for idx in top_3_indices]
    
    # Display the top 1 prediction
    predicted_class, confidence = top_3_predictions[0]
    
    # Show result
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Display the second and third most likely predictions
    st.subheader("Other likely breeds:")
    for breed, conf in top_3_predictions[1:]:
        st.write(f"{breed}: {conf:.2f}%")