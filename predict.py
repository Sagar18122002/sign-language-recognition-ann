import numpy as np
import joblib
import tensorflow as tf


# Load saved objects
model = tf.keras.models.load_model("sign_language_ann.keras")

scaler = joblib.load("scaler.pkl")

label_encoder = joblib.load("label_encoder.pkl")


def extract_features(hand_landmarks):

    features = []

    for landmark in hand_landmarks:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(features, dtype=np.float32)
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
    predicted_class = np.argmax(probabilities[0])

    # Convert numerical class back to sign
    predicted_label = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    # Confidence
    confidence = probabilities[0][predicted_class]

    return predicted_label, confidence