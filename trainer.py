import matplotlib.pyplot as plt


def train_model(model,train_gen,test_gen,epochs,save_path):
    history = model.fit(
        train_gen,
        epochs = epochs,
        validation_data=test_gen
    )
    model.save("Waste_Classifier.keras")
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()