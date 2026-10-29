
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Data" / "ravdess"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "emotion_features.csv"

# --------------------------------------------------
# 2. Emotion mapping for RAVDESS speech files
# --------------------------------------------------

EMOTIONS = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}

SAMPLE_RATE = 16000
N_MFCC = 40
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512


# --------------------------------------------------
# 3. Helper: summarize feature values over time
# --------------------------------------------------

def summarize(feature_matrix):
    """Return mean and standard deviation for each feature."""
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)

    mean_values = np.mean(feature_matrix, axis=1)
    std_values = np.std(feature_matrix, axis=1)

    return np.concatenate([mean_values, std_values])


# --------------------------------------------------
# 4. Extract enhanced features from one audio file
# --------------------------------------------------

def extract_features(audio_path):
    audio, sr = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) < N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio)))

    # Normalize volume to reduce recording-level differences.
    audio = librosa.util.normalize(audio)

    features = {}

    # A. MFCC: voice timbre
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    for i, value in enumerate(np.mean(mfcc, axis=1)):
        features[f"mfcc_mean_{i + 1}"] = value

    for i, value in enumerate(np.std(mfcc, axis=1)):
        features[f"mfcc_std_{i + 1}"] = value

    # B. Delta MFCC: changes in speech
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for prefix, matrix in [
        ("delta", delta),
        ("delta2", delta2),
    ]:
        for i, value in enumerate(np.mean(matrix, axis=1)):
            features[f"{prefix}_mean_{i + 1}"] = value

        for i, value in enumerate(np.std(matrix, axis=1)):
            features[f"{prefix}_std_{i + 1}"] = value

    # C. Chroma: pitch-class energy
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    for i, value in enumerate(np.mean(chroma, axis=1)):
        features[f"chroma_mean_{i + 1}"] = value

    for i, value in enumerate(np.std(chroma, axis=1)):
        features[f"chroma_std_{i + 1}"] = value

    # D. Mel spectrogram: energy across frequency bands
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    for i, value in enumerate(np.mean(mel_db, axis=1)):
        features[f"mel_mean_{i + 1}"] = value

    for i, value in enumerate(np.std(mel_db, axis=1)):
        features[f"mel_std_{i + 1}"] = value

    # E. Spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    for i, value in enumerate(np.mean(contrast, axis=1)):
        features[f"contrast_mean_{i + 1}"] = value

    for i, value in enumerate(np.std(contrast, axis=1)):
        features[f"contrast_std_{i + 1}"] = value

    # F. Spectral characteristics
    spectral_features = {
        "centroid": librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=N_FFT,
            hop_length=HOP_LENGTH
        ),
        "bandwidth": librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=N_FFT,
            hop_length=HOP_LENGTH
        ),
        "rolloff": librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=N_FFT,
            hop_length=HOP_LENGTH
        ),
        "zcr": librosa.feature.zero_crossing_rate(
            audio, frame_length=N_FFT,
            hop_length=HOP_LENGTH
        ),
        "rms": librosa.feature.rms(
            y=audio, frame_length=N_FFT,
            hop_length=HOP_LENGTH
        ),
    }

    for name, matrix in spectral_features.items():
        features[f"{name}_mean"] = float(np.mean(matrix))
        features[f"{name}_std"] = float(np.std(matrix))

    # G. Fundamental frequency (pitch)
    pitch = librosa.yin(
        audio,
        fmin=65,
        fmax=400,
        sr=sr,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Log pitch reduces the effect of large absolute differences.
    log_pitch = np.log(np.maximum(pitch, 1e-6))

    features["pitch_mean"] = float(np.mean(log_pitch))
    features["pitch_std"] = float(np.std(log_pitch))
    features["pitch_min"] = float(np.min(log_pitch))
    features["pitch_max"] = float(np.max(log_pitch))

    # Prevent invalid values from reaching the model.
    result = np.asarray(list(features.values()), dtype=np.float64)
    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return dict(zip(features.keys(), result))


# --------------------------------------------------
# 5. Process the RAVDESS speech dataset
# --------------------------------------------------

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}\n"
        "Update DATASET_DIR to your actual RAVDESS folder."
    )

audio_files = sorted(DATASET_DIR.rglob("*.wav"))

if not audio_files:
    raise FileNotFoundError(
        f"No WAV files found in: {DATASET_DIR}"
    )

print(f"Found {len(audio_files)} WAV files.")

rows = []
skipped = 0

for index, audio_path in enumerate(audio_files, start=1):
    parts = audio_path.stem.split("-")

    # RAVDESS speech-only recordings
    if len(parts) != 7:
        continue

    if parts[0] != "03" or parts[1] != "01":
        continue

    emotion_code = parts[2]

    if emotion_code not in EMOTIONS:
        continue

    try:
        row = extract_features(audio_path)

        row["emotion"] = EMOTIONS[emotion_code]
        row["actor_id"] = int(parts[6])

        rows.append(row)

    except Exception as error:
        skipped += 1
        print(f"Skipped {audio_path.name}: {error}")

    if index % 100 == 0:
        print(f"Checked {index}/{len(audio_files)} files")

# --------------------------------------------------
# 6. Save the extracted features
# --------------------------------------------------

if not rows:
    raise RuntimeError(
        "No features were extracted. Check the dataset folder "
        "and RAVDESS filename format."
    )

df = pd.DataFrame(rows)

feature_columns = [
    column for column in df.columns
    if column not in ("emotion", "actor_id")
]

df[feature_columns] = (
    df[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

df.to_csv(OUTPUT_FILE, index=False)

print("\nEnhanced feature extraction completed!")
print("Dataset shape:", df.shape)
print("Number of audio samples:", len(df))
print("Features per sample:", len(feature_columns))
print("\nEmotion counts:")
print(df["emotion"].value_counts())
print("\nActors:", df["actor_id"].nunique())
print("Skipped audio files:", skipped)
print("Saved to:", OUTPUT_FILE)
