
import os
_KERAS_CACHE = str(__import__("pathlib").Path(__file__).resolve().parent.parent / ".keras_cache")
os.environ["KERAS_HOME"] = _KERAS_CACHE
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Support both 'dataset' (lowercase) and 'Dataset' folder names
DATASET_DIR = ROOT / "dataset" if (ROOT / "dataset").exists() else ROOT / "Dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_valid_images(split):
    """Scan a split folder and return list of (path, label_int) for valid images.
    
    We skip PIL.verify() here — TF will silently skip corrupt images during
    decode, which is much faster on large/network-mounted datasets.
    """
    records = []
    split_dir = DATASET_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Folder not found: {split_dir}")

    for label, letter in enumerate(CLASSES):
        class_dir = split_dir / letter
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                records.append((str(path), label))

    if not records:
        raise RuntimeError(f"No valid images found in {split_dir}")

    print(f"{split}: {len(records)} images across {len(CLASSES)} classes")
    return records


def load_image(path, label):
    """Load, resize and apply MobileNetV2 preprocessing to one image."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def make_dataset(split, training=False):
    """Build a tf.data pipeline for a given split."""
    records = get_valid_images(split)
    paths, labels = zip(*records)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    if training:
        ds = ds.shuffle(len(records), reshuffle_each_iteration=True)

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        # Augmentation: rotation, zoom, translation, flip, brightness, contrast
        # Flip helps with S vs Z confusion (mirrored hand shapes)
        def augment(image, label):
            # MobileNetV2 input is [-1, 1]; convert to [0, 1] for color augments
            image_01 = (image + 1.0) / 2.0
            image_01 = tf.image.random_brightness(image_01, max_delta=0.15)
            image_01 = tf.image.random_contrast(image_01, lower=0.8, upper=1.2)
            image_01 = tf.clip_by_value(image_01, 0.0, 1.0)
            # Back to [-1, 1]
            image = image_01 * 2.0 - 1.0
            image = tf.image.random_flip_left_right(image)
            return image, label

        geometric = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ])

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(
            lambda x, y: (geometric(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    for split in ["Training", "Validation", "Testing"]:
        ds = make_dataset(split, training=(split == "Training"))
        print(f"{split} dataset ready.")
