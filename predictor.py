import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def predict_image(model_path, img_path):
    model = load_model(model_path)  # Load SavedModel folder

    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        return f"Organic ({prediction:.2f})"
    else:
        return f"Recyclable ({prediction:.2f})"
