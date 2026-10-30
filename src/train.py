
import os
_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
os.environ["KERAS_HOME"] = _ROOT + "/.keras_cache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import tensorflow as tf
from pathlib import Path
from data import make_dataset
from models import build_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "isl_model.keras"

tf.keras.utils.set_random_seed(42)

# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading datasets...")
train_ds = make_dataset("Training",   training=True)
val_ds   = make_dataset("Validation", training=False)

# ── Build model ────────────────────────────────────────────────────────────────
model, base = build_model()

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Train only the new head (base frozen)
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== Phase 1: Training head (base frozen) ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase1 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max",
        patience=4, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=2, min_lr=1e-6, verbose=1
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    callbacks=callbacks_phase1
)

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Fine-tune top layers of MobileNetV2
# Unfreeze from layer 100 onward (keeps early spatial features frozen).
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== Phase 2: Fine-tuning top MobileNetV2 layers ===")
base.trainable = True
for layer in base.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower LR for fine-tune
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max",
        patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3,
        patience=2, min_lr=1e-7, verbose=1
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks_phase2
)

model.save(MODEL_PATH)
print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")
