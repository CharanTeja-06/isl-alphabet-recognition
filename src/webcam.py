
import os
_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
os.environ["KERAS_HOME"] = _ROOT + "/.keras_cache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque, Counter

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "isl_model.keras"

CLASSES             = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMG_SIZE            = 224
CONFIDENCE_THRESHOLD = 0.60   # Reject uncertain predictions
HISTORY_LEN         = 10      # Frames kept for voting
STABLE_FRAMES       = 15      # Hold this long to confirm a letter

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model…")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded. Opening webcam…")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not available.")

history        = deque(maxlen=HISTORY_LEN)
candidate      = None
candidate_frames = 0
last_added     = None
text_buffer    = ""


def detect_hand_roi(frame):
    """
    Simple skin-colour hand detector using YCrCb thresholding.
    Returns (has_hand: bool, roi_rect: (x,y,w,h) or None).
    The ROI is the bounding box of the largest skin blob.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb,
                       np.array([0,  133, 77],  np.uint8),
                       np.array([255, 173, 127], np.uint8))
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    h, w = frame.shape[:2]
    min_area = 0.01 * h * w   # Hand must be at least 1% of frame

    if area < min_area:
        return False, None

    x, y, bw, bh = cv2.boundingRect(largest)
    # Add small padding
    pad = 20
    x = max(0, x - pad);  y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad);  bh = min(h - y, bh + 2 * pad)
    return True, (x, y, bw, bh)


def preprocess(frame_rgb):
    """Resize + MobileNetV2 preprocess for model input."""
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(
        img.astype(np.float32)
    )
    return img


try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)   # Mirror for natural feel
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        has_hand, roi = detect_hand_roi(frame)

        if has_hand and roi is not None:
            x, y, bw, bh = roi
            # Draw hand bounding box in green
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            # Crop and classify the hand region
            hand_rgb = rgb[y:y + bh, x:x + bw]
            img_input = preprocess(hand_rgb)
        else:
            # No hand — classify the whole frame but mark as no-hand
            img_input = preprocess(rgb)

        probabilities = model.predict(img_input[None, ...], verbose=0)[0]
        idx        = int(np.argmax(probabilities))
        confidence = float(probabilities[idx])

        # Temporal voting
        if has_hand and confidence >= CONFIDENCE_THRESHOLD:
            history.append(idx)
        else:
            history.append(-1)

        valid = [x for x in history if x >= 0]
        if valid:
            stable_idx, votes = Counter(valid).most_common(1)[0]
            stable_letter = CLASSES[stable_idx]
        else:
            stable_idx, votes = -1, 0
            stable_letter = "Uncertain"

        # Hold-to-confirm logic
        if has_hand and confidence >= CONFIDENCE_THRESHOLD and stable_idx >= 0:
            if stable_idx == candidate:
                candidate_frames += 1
            else:
                candidate = stable_idx
                candidate_frames = 1

            if candidate_frames >= STABLE_FRAMES and candidate != last_added:
                text_buffer += CLASSES[candidate]
                last_added = candidate
        else:
            candidate       = None
            candidate_frames = 0
            if not has_hand:
                last_added = None   # Allow re-adding after hand leaves

        # ── Progress bar toward confirmation ──────────────────────────────────
        if has_hand and candidate is not None:
            bar_w  = 200
            filled = int(bar_w * min(candidate_frames / STABLE_FRAMES, 1.0))
            cv2.rectangle(frame, (15, 125), (15 + bar_w, 145), (80, 80, 80), -1)
            cv2.rectangle(frame, (15, 125), (15 + filled, 145), (0, 200, 255), -1)
            cv2.putText(frame, "Hold to confirm", (220, 142),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # ── HUD overlays ───────────────────────────────────────────────────────
        # Status line (green = has hand, red = no hand)
        if has_hand:
            status_color = (0, 255, 0)
            status_text  = f"Prediction: {stable_letter}  ({confidence:.0%})"
        else:
            status_color = (0, 0, 255)
            status_text  = "No hand detected"

        cv2.putText(frame, status_text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

        # Top-3 predictions
        top3     = np.argsort(probabilities)[-3:][::-1]
        top_text = "  ".join(
            f"{CLASSES[i]}:{probabilities[i]:.0%}" for i in top3
        )
        cv2.putText(frame, top_text, (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # Typed text
        cv2.putText(frame, f"Text: {text_buffer}", (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Key hints at bottom
        cv2.putText(frame, "Q: quit | C: clear | B: backspace",
                    (15, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("ISL Alphabet Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            text_buffer = ""
            last_added  = None
        elif key == ord("b"):
            text_buffer = text_buffer[:-1]
            last_added  = None

finally:
    cap.release()
    cv2.destroyAllWindows()
