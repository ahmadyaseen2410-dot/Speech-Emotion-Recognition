from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from extractfeatures import extract_features


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "speech_emotion_ann.keras"
SCALER_PATH = MODEL_DIR / "ann_scaler.joblib"
ENCODER_PATH = MODEL_DIR / "ann_label_encoder.joblib"
COLUMNS_PATH = MODEL_DIR / "ann_feature_columns.joblib"


# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Speech Emotion Recognition")
st.write(
    "Upload a speech audio file to predict its emotion "
    "using a trained Artificial Neural Network (ANN)."
)

st.info("Supported emotions: Happy, Sad, Angry, Neutral")


# Check required files
required_files = [
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    COLUMNS_PATH,
]

missing_files = [
    str(path.name)
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.error(
        "Missing model files: " + ", ".join(missing_files)
    )
    st.stop()


# Load model and preprocessing objects
@st.cache_resource
def load_model_files():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)

    return model, scaler, encoder, feature_columns


try:
    model, scaler, encoder, feature_columns = load_model_files()
    st.success("ANN model loaded successfully.")

except Exception as error:
    st.error(f"Error loading model files: {error}")
    st.stop()


# Upload audio
uploaded_file = st.file_uploader(
    "Upload a speech audio file",
    type=["wav"],
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Predict Emotion", type="primary"):
        temp_path = None

        try:
            with st.spinner("Analyzing speech..."):

                # Save uploaded audio temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav",
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = Path(temp_file.name)

                # Extract audio features
                features = extract_features(str(temp_path))

                # Prepare feature DataFrame
                if isinstance(features, dict):
                    feature_df = pd.DataFrame([features])
                else:
                    feature_df = pd.DataFrame(
                        [np.asarray(features).flatten()],
                        columns=feature_columns,
                    )

                # Match the training feature order
                feature_df = feature_df.reindex(
                    columns=feature_columns,
                    fill_value=0,
                )

                X = feature_df.to_numpy(dtype=np.float32)

                # Apply the scaler used during training
                X_scaled = scaler.transform(X)

                # ANN prediction
                scores = np.asarray(
                    model.predict(X_scaled, verbose=0)
                )[0]

                predicted_index = int(np.argmax(scores))

                # Decode predicted class
                emotion = encoder.inverse_transform(
                    np.array([predicted_index])
                )[0]

            # Display result
            st.success("Prediction completed!")
            st.subheader(
                f"Predicted Emotion: {str(emotion).upper()}"
            )

            messages = {
                "happy": "The model predicts happy speech.",
                "sad": "The model predicts sad speech.",
                "angry": "The model predicts angry speech.",
                "neutral": "The model predicts neutral speech.",
            }

            st.write(
                messages.get(
                    str(emotion).lower(),
                    "The model returned a prediction.",
                )
            )

            # Display output scores
            st.subheader("Emotion Scores")

            score_data = {
                str(label).capitalize(): float(scores[i])
                for i, label in enumerate(encoder.classes_)
            }

            st.bar_chart(score_data)

            st.caption(
                "These scores are ANN outputs and may not be "
                "calibrated probabilities. Predictions are not "
                "definitive assessments of a person's emotions."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)