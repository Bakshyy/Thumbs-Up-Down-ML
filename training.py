import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib

DATA_DIR = pathlib.Path("data")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode="categorical"
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode="categorical"
)

# Normalize
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

# CNN model
model = keras.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=IMG_SIZE + (3,)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(2, activation="softmax")  # two classes: thumbs up / down
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

model.save("thumbs_model.keras")
