from keras.models import load_model
from predictor import predict_image
import os

MODEL_PATH = "Waste_Classifier.keras"
SAMPLE_DIR = r"C:\waste\DATASET\SAMPLE"

model = load_model(MODEL_PATH)

for img_file in os.listdir(SAMPLE_DIR):
    img_path = os.path.join(SAMPLE_DIR, img_file)
    result = predict_image(MODEL_PATH, img_path)
    print('Prediction:', result)