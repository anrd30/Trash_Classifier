from data_loader import get_data_loader
from model import create_model
from trainer import train_model
from predictor import predict_image
import os

TRAIN_DIR = r"C:\waste\DATASET\TRAIN"
TEST_DIR = r"C:\waste\DATASET\TEST"
MODEL_PATH = "Waste_Classifier.keras"
SAMPLE_DIR = r"C:\waste\DATASET\SAMPLE"
if __name__ == "__main__":

    train_gen, test_gen = get_data_loader(TRAIN_DIR,TEST_DIR)

    model = create_model()
    
    train_model(model, train_gen, test_gen, epochs=10, save_path=MODEL_PATH)

    for img_file in os.listdir(SAMPLE_DIR):
        img_path = os.path.join(SAMPLE_DIR, img_file)
        result = predict_image(MODEL_PATH, img_path)
        print('Prediction:', result)