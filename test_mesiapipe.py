import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------
# Load trained model and objects
# --------------------------------

model = tf.keras.models.load_model(
    "sign_language_ann.keras"
)

scaler = joblib.load(
    "scaler.pkl"
)

label_encoder = joblib.load(
    "label_encoder.pkl"
)


# --------------------------------
# MediaPipe setup
# --------------------------------

MODEL_PATH = "models/hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# --------------------------------
# Feature Extraction
# --------------------------------

def extract_features(hand_landmarks):

    features = []

    for landmark in hand_landmarks:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(
        features,
        dtype=np.float32
    )


# --------------------------------
# Prediction Function
# --------------------------------

def predict_sign(features):

    # Convert to 2D array
    features = features.reshape(1, -1)

    # Apply the SAME scaler used during training
    features_scaled = scaler.transform(features)

    # Get probabilities
    probabilities = model.predict(
        features_scaled,
        verbose=0
    )

    # Get class with highest probability
    predicted_class = np.argmax(
        probabilities[0]
    )

    # Convert numerical class back to sign
    predicted_label = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    # Confidence
    confidence = probabilities[0][predicted_class]

    return predicted_label, confidence


# --------------------------------
# Webcam
# --------------------------------

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Extract 63 features
        features = extract_features(hand)

        # Predict sign
        if len(features) == 63:

            label, confidence = predict_sign(
                features
            )

            text = f"{label} ({confidence * 100:.2f}%)"

            cv2.putText(
                frame,
                text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Sign Language Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()