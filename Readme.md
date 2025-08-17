# Waste Classifier

A deep learning project for classifying waste images using Keras and TensorFlow.

## Features

- Train a custom image classifier on waste categories
- Predict waste type from new images
- Modular code for data loading, training, and prediction

## Dataset

This project uses the [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data?resource=download) from Kaggle.

**To use this project:**
1. Download the dataset from Kaggle.
2. Place it in the `DATASET/` folder in the project root.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/anrd30/Trash_Classifier
   cd waste-classifier
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv waste-venv
   waste-venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Training

To train the model:
```
python main.py
```

## Prediction

To predict using a saved model:
```
python predict_only.py
```

## File Structure

- `main.py` — Train and predict
- `trainer.py` — Training utilities
- `predictor.py` — Prediction utilities
- `data_loader.py` — Data loading utilities
- `predict_only.py` — Predict without retraining

## License

[MIT](LICENSE)
