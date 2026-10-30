
import os
_KERAS_CACHE = str(__import__("pathlib").Path(__file__).resolve().parent.parent / ".keras_cache")
os.environ["KERAS_HOME"] = _KERAS_CACHE
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

IMG_SIZE = 224
NUM_CLASSES = 26


def build_model(fine_tune_from=100):
    """
    MobileNetV2-based ISL classifier.

    Phase 1: base frozen — only the new head trains.
    Phase 2: layers from fine_tune_from onward are unfrozen for fine-tuning.

    The head: GlobalAvgPool → Dense(256, relu) → Dropout(0.4) → Dense(26, softmax)
    The extra Dense layer gives the model more capacity to separate similar gestures.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # Start frozen; unfreeze in train.py phase 2

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", dtype="float32"
    )(x)

    return tf.keras.Model(inputs, outputs), base


if __name__ == "__main__":
    model, _ = build_model()
    model.summary()
