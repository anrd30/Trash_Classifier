import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_loader(train_dir,test_dir,img_size=(128,128),batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow_from_directory(
        directory=train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
    )
    test_generator = datagen.flow_from_directory(
        directory=test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    return train_generator, test_generator