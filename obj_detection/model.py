import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import numpy as np


class SSD_Model:
    def __init__(
        self,
        num_classes: int,
        input_size: tuple,
        optimizer: any = "adam",
        loss: any = ["categorical_crossentropy", "mse"],
    ) -> None:
        input_images = layers.Input(shape=input_size, name="input_images")
        input_bboxes = layers.Input(shape=(4,), name="input_bboxes")

        # Base VGG16 model
        base_model = VGG16(
            weights="imagenet", include_top=False, input_tensor=input_images
        )
        base_out = base_model.output
        x = layers.Flatten()(base_out)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        predictions = layers.Dense(
            4, activation="sigmoid", name="predictions"
        )(x)

        predictions_class = layers.Dense(
            num_classes, activation="softmax", name="predictions_class"
        )(x)

        model = Model(
            inputs=[input_images, input_bboxes],
            outputs=[predictions, predictions_class],
        )
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        self.model = model

    def model_fit(
        self, train_images, train_labels, val_images, val_labels, batch_size
    ):
        train_bboxes = np.array([label["bbox"] for label in train_labels])
        train_labels = np.array([label["label"] for label in train_labels])

        val_bboxes = np.array([label["bbox"] for label in val_labels])
        val_labels = np.array([label["label"] for label in val_labels])

        self.model.fit(
            x=[np.array(train_images), train_bboxes],
            y=[
                train_bboxes,
                train_labels,
            ],
            epochs=20,
            validation_data=(
                [np.array(val_images), val_bboxes],
                [val_bboxes, val_labels],
            ),
            steps_per_epoch=len(train_images) // batch_size,
            validation_steps=len(val_images) // batch_size,
        )
