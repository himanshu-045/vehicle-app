import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import cv2
import numpy as np
from PIL import Image

# --- Load the Saved Model and Label Encoder ---
# Make sure these files are in the same folder as your app.py script
model = load_model('vehicle_model.h5')
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)
# ---------------------------------------------

st.title("Vehicle Classification App 🚗")
st.write("Upload an image of a vehicle, and the model will predict its type.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the model
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (100, 100))
    img_processed = np.expand_dims(img_resized, axis=0) / 255.0

    # Make a prediction
    prediction = model.predict(img_processed)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = le.inverse_transform([predicted_class_index])

    # Display the result
    st.success(f"The model predicts this is a: **{predicted_class_name[0]}**")