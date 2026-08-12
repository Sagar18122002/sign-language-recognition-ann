import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import mediapipe as mp

from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load trained ANN
model = tf.keras.models.load_model(
    "sign_language_ann.keras"
)

# Load scaler
scaler = joblib.load(
    "scaler.pkl"
)

# Load label encoder
label_encoder = joblib.load(
    "label_encoder.pkl"
)

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
def predict_sign(features):

    features = features.reshape(1, -1)

    features_scaled = scaler.transform(
        features
    )

    probabilities = model.predict(
        features_scaled,
        verbose=0
    )

    predicted_class = np.argmax(
        probabilities[0]
    )

    predicted_label = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = probabilities[0][
        predicted_class
    ]

    return predicted_label, confidence
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language Recognition")
st.write(
    "Real-time sign language recognition "
    "using MediaPipe and Artificial Neural Network"
)

st.divider()

# Sidebar
st.sidebar.title("Options")

option = st.sidebar.radio(
    "Choose Input Method",
    ["Upload Image", "Webcam"]
)

if option == "Upload Image":

    st.subheader("📷 Upload a Hand Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="upload_image"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        rgb_image = np.array(image)

        st.image(
            rgb_image,
            caption="Uploaded Image",
            use_container_width=True
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = detector.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            features = extract_features(hand)

            if len(features) == 63:

                label, confidence = predict_sign(features)

                st.success(
                    f"Predicted Sign: **{label}**"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

        else:
            st.warning(
                "No hand detected. Please upload a clear hand image."
            )


elif option == "Webcam":

    st.subheader("📸 Take a Picture Using Webcam")

    camera_image = st.camera_input(
        "Show your hand to the camera",
        key="webcam_input"
    )

    if camera_image is not None:

        image = Image.open(camera_image).convert("RGB")
        rgb_image = np.array(image)

        st.image(
            rgb_image,
            caption="Captured Image",
            use_container_width=True
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = detector.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            features = extract_features(hand)

            if len(features) == 63:

                label, confidence = predict_sign(features)

                st.success(
                    f"Predicted Sign: **{label}**"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

        else:
            st.warning(
                "No hand detected. Please show your hand clearly.")
