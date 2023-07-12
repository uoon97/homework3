import os
import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

def load_data():
    train_dir = os.path.join(os.getcwd(), '100_1')
    val_dir = os.path.join(os.getcwd(), '100_2')

    types = ['train', 'val']
    numbers = range(1, 10)

    for t in types:
        for i in numbers:
            globals()[f'{t}_{i}_fname'] = os.listdir(os.path.join(globals()[f'{t}_dir'], i))

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for t in types:
        for n in numbers:
                for i in range(len(globals()[f'{t}_{n}_fname'])):
                    f = os.path.join(globals()[f'{t}_dir'], n, globals()[f'{t}_{n}_fname'][i])
                    image = PIL.Image.open(f)
                    image = image.resize((224, 224))
                    arr = np.array(image)
                    globals()[f'X_{t}'].append(arr)
                    globals()[f'y_{t}'].append(n)

        X_train = np.array(X_train)/255.
        y_train = np.array(y_train)
        X_val = np.array(X_val)/255.
        y_val = np.array(y_val)

    return X_train, y_train, X_val, y_val

def fit_model(X_train, y_train, X_val, y_val):
    model = keras.Sequential([
        layers.Conv2D(128, 3, activation = 'relu', input_shape = X_train.shape[1:]),
        layers.Conv2D(64, 3, activation = 'relu'),
        layers.MaxPooling2D(pool_size = (2, 2)),

        layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'),
        layers.Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu'),
        layers.MaxPooling2D(pool_size = (2, 2)),

        layers.Flatten(),

        layers.Dense(units = 256, activation = 'relu'),
        layers.Dense(units = 1, activation = 'sigmoid'),
    ])

    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
    )

    es = EarlyStopping(monitor = 'val_loss', patience = 10)
    ckpt = ModelCheckpoint(
        '/ckpt', monitor = 'val_loss', 
        save_weights_only = True,
        save_best_only = True
        )

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs = 30,
        batch_size = 32,
        callbacks = [es, ckpt]
    )