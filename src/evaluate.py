
import os
_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
os.environ["KERAS_HOME"] = _ROOT + "/.keras_cache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf
from pathlib import Path
from data import make_dataset, CLASSES

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "isl_model.keras"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train.py first.")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading test set...")
test_ds = make_dataset("Testing", training=False)

# ── Overall accuracy ───────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest accuracy : {accuracy * 100:.2f}%")
print(f"Test loss     : {loss:.4f}")

# ── Per-letter accuracy + confusion matrix ─────────────────────────────────────
correct = [0] * 26
total   = [0] * 26
confusion = np.zeros((26, 26), dtype=int)

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    pred_cls = np.argmax(preds, axis=1)
    label_arr = labels.numpy()

    for true, pred in zip(label_arr, pred_cls):
        total[true]   += 1
        correct[true] += int(true == pred)
        confusion[true][pred] += 1

print("\n-- Per-letter accuracy --------------------------------")
low_letters = []
for i, letter in enumerate(CLASSES):
    if total[i]:
        acc = 100 * correct[i] / total[i]
        flag = "  [OK]" if acc >= 80 else "  [LOW]"
        print(f"  {letter}: {acc:6.2f}%{flag}")
        if acc < 80:
            low_letters.append(letter)

if low_letters:
    print(f"\nLetters below 80%: {', '.join(low_letters)}")
else:
    print("\nAll letters are above 80% accuracy.")

# -- Confusion matrix (text) ----------------------------------------------------
print("\n-- Confusion matrix (rows=true, cols=predicted) ------")
header = "   " + " ".join(f"{c:>3}" for c in CLASSES)
print(header)
for i, letter in enumerate(CLASSES):
    row = f"{letter}  " + " ".join(f"{confusion[i][j]:>3}" for j in range(26))
    print(row)

# -- Worst confusions ----------------------------------------------------------
print("\n-- Top-10 most confused pairs ------------------------")
pairs = []
for i in range(26):
    for j in range(26):
        if i != j and confusion[i][j] > 0:
            pairs.append((confusion[i][j], CLASSES[i], CLASSES[j]))
pairs.sort(reverse=True)
for count, true_l, pred_l in pairs[:10]:
    print(f"  {true_l} → predicted as {pred_l}: {count} times")
