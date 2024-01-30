import tensorflow as tf

from .utils import get_model_tag


def LeNet5(input_shape, n_classes):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                6,
                kernel_size=5,
                strides=1,
                activation="tanh",
                input_shape=input_shape,
                padding="same",
            ),
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2), padding="valid"
            ),
            tf.keras.layers.Conv2D(
                16, kernel_size=5, strides=1, activation="tanh", padding="valid"
            ),
            tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2), padding="valid"
            ),
            tf.keras.layers.Conv2D(
                120, kernel_size=5, strides=1, activation="tanh", padding="valid"
            ),
            tf.keras.layers.Flatten(),  # Flatten
            tf.keras.layers.Dense(84, activation="tanh"),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.tag = "LeNet-5"
    return model


def CNN1(input_shape, n_classes):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=1,
                activation="relu",
                input_shape=input_shape,
            ),
            tf.keras.layers.MaxPooling2D((2, 2), strides=1),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.tag = get_model_tag(model)
    return model


def CNN2(input_shape, n_classes):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(5, 5),
                strides=2,
                activation="relu",
                input_shape=input_shape,
            ),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=1,
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.tag = get_model_tag(model)
    return model


def CNN3(input_shape, n_classes):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(7, 7),
                strides=1,
                activation="relu",
                input_shape=input_shape,
            ),
            tf.keras.layers.MaxPooling2D((2, 2), strides=1),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(5, 5),
                strides=1,
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D((2, 2), strides=1),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=1,
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D((2, 2), strides=1),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.tag = get_model_tag(model)
    return model


def load_architecture(
    base_model,
    input_shape,
    n_classes,
    input_model_shape=None,
    rescale_args={"scale": 1},
    **kwargs,
):
    assert input_shape[-1] == 3, "Base architectures work with 3 channels only"

    resize = True
    if input_model_shape is None:
        input_model_shape = input_shape
        resize = False

    model_instance = base_model(
        # weights="imagenet",  # Load weights pre-trained on ImageNet.
        weights=None,
        input_shape=input_model_shape,
        include_top=False,
        **kwargs,
    )

    # Define the model
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Rescaling(**rescale_args)(inputs)
    if resize:
        x = tf.keras.layers.Resizing(*input_model_shape[:2])(x)
    x = model_instance(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.tag = get_model_tag(model)
    return model


def EfficientNetB0(input_shape, n_classes):
    return load_architecture(
        tf.keras.applications.EfficientNetB0,
        input_shape,
        n_classes,
        input_model_shape=None,
        rescale_args={"scale": 255},
    )


def InceptionV3(input_shape, n_classes):
    return load_architecture(
        tf.keras.applications.InceptionV3,
        input_shape,
        n_classes,
        input_model_shape=(128, 128),
        rescale_args={"scale": 2, "offset": -1},
    )


def Xception(input_shape, n_classes):
    return load_architecture(
        tf.keras.applications.Xception,
        input_shape,
        n_classes,
        input_model_shape=(128, 128),
        rescale_args={"scale": 2, "offset": -1},
    )


def ResNet50V2(input_shape, n_classes):
    return load_architecture(
        tf.keras.applications.ResNet50V2,
        input_shape,
        n_classes,
        input_model_shape=None,
        rescale_args={"scale": 2, "offset": -1},
    )


def InceptionResNetV2(input_shape, n_classes):
    return load_architecture(
        tf.keras.applications.InceptionResNetV2,
        input_shape,
        n_classes,
        input_model_shape=(128, 128, 3),
        rescale_args={"scale": 2, "offset": -1},
    )


def VGG16(input_shape, n_classes):
    return load_architecture(
        tf.keras.applications.VGG16,
        input_shape,
        n_classes,
        input_model_shape=None,
        rescale_args={"scale": 2, "offset": -1},
    )
