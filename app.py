import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import cv2
import numpy as np
from PIL import Image
import requests
import os

# --- Function to download files from Google Drive ---
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


MODEL_ID = "https://drive.google.com/file/d/1L0uKHtHZcdRkfj7q_RJexSY88-1I0FzM/view?usp=sharing"
ENCODER_ID = "https://drive.google.com/file/d/1GPsCS4HfX6zqQSVd8baVZsG496XIS8k2/view?usp=drive_link"
# -----------------------------------------

# --- Download files if they don't exist ---
if not os.path.exists('vehicle_model.h5'):
    with st.spinner("Downloading model... this may take a minute."):
        download_file_from_google_drive(MODEL_ID, 'vehicle_model.h5')

if not os.path.exists('label_encoder.pkl'):
     with st.spinner("Downloading label encoder..."):
        download_file_from_google_drive(ENCODER_ID, 'label_encoder.pkl')
# ------------------------------------------


# --- Load the Saved Model and Label Encoder ---
model = load_model('vehicle_model.h5')
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)
# ---------------------------------------------

st.title("Vehicle Classification App 🚗")
st.write("Upload an image of a vehicle, and the model will predict its type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (100, 100))
    img_processed = np.expand_dims(img_resized, axis=0) / 255.0

    prediction = model.predict(img_processed)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = le.inverse_transform([predicted_class_index])

    st.success(f"The model predicts this is a: **{predicted_class_name[0]}**")
