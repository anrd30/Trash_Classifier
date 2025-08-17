import streamlit as st
from keras.models import load_model
from predictor import predict_image
import tempfile

MODEL_PATH = "Waste_Classifier.keras"

st.title("Waste Classifier")
st.write("Upload a photo of waste to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        img_path = tmp_file.name

    st.image(img_path, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        result = predict_image(MODEL_PATH, img_path)
        st.success(f"Prediction: {result}")