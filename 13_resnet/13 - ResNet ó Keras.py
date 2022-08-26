from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
import matplotlib.pyplot as plt
import numpy as np
 
 
def load_train(path):

    train_datagen = ImageDataGenerator(rescale=1/255.)

    train_datagen_flow = train_datagen.flow_from_directory(path, target_size=(150, 150), batch_size=16, class_mode='sparse', seed=123456)

    return train_datagen_flow


def create_model(input_shape):

    optimizer = Adam(lr=0.0001)

    backbone = ResNet50(input_shape=input_shape, weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False) 

    model = Sequential()

    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(12, activation='softmax')) 

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc']) 

    return model 

 
def train_model(model, train_data, test_data, epochs=4, steps_per_epoch=None, validation_steps=None, batch_size=0):
    if steps_per_epoch is None:
        steps_per_epoch=len(train_data)
 
    if validation_steps is None:
        validation_steps=len(test_data)
 
    model.fit(train_data, validation_data=test_data, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, verbose=2, epochs=epochs)
 
    return model