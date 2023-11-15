from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout


class SSD_Model:
    def __init__(
        self,
        num_classes: int,
        input_size: tuple,
        optimizer: any = "adam",
        losses: any = ["categorical_crossentropy", "mean_squared_error"],
    ) -> None:
        input_images = Input(shape=input_size, name="input_images")

        base_model = VGG16(
            weights="imagenet", include_top=False, input_tensor=input_images
        )
        base_model.trainable = False
        base_out = base_model.output
        flatten_output = Flatten()(base_out)
        bbox_layers = Dense(128, activation="relu")(flatten_output)
        bbox_layers = Dense(64, activation="relu")(bbox_layers)
        bbox_layers = Dropout(0.2)(bbox_layers)
        bbox_layers = Dense(32, activation="relu")(bbox_layers)
        bbox_layers = Dense(16, activation="relu")(bbox_layers)
        bbox_layers = Dropout(0.2)(bbox_layers)

        label_layers = Dense(512, activation="relu")(flatten_output)
        label_layers = Dense(512, activation="relu")(label_layers)
        label_layers = Dropout(0.2)(label_layers)
        label_layers = Dense(256, activation="relu")(label_layers)
        label_layers = Dense(256, activation="relu")(label_layers)
        label_layers = Dropout(0.2)(label_layers)

        bounding_box = Dense(4, activation="sigmoid", name="bounding_box")(
            bbox_layers
        )

        predictions_class = Dense(
            num_classes, activation="softmax", name="class_label"
        )(label_layers)

        model = Model(
            inputs=base_model.input,
            outputs=(bounding_box, predictions_class),
        )
        model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=["accuracy"],
        )
        model.summary()
        self.model = model

    def model_fit(
        self,
        train_images,
        train_targets,
        val_images,
        val_targets,
        batch_size,
        epochs,
    ):
        self.model.fit(
            train_images,
            train_targets,
            validation_data=(
                val_images,
                val_targets,
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
        self.model.save("../models/sign_dectection_model.pb")
