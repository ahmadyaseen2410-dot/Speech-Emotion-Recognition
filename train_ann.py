
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
)

# --------------------------------------------------
# 1. Paths and reproducibility
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "outputs" / "emotion_features.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Run extractfeatures.py first."
    )

# --------------------------------------------------
# 2. Load features
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

if not {"emotion", "actor_id"}.issubset(df.columns):
    raise ValueError("CSV must contain emotion and actor_id columns.")

feature_columns = [
    column for column in df.columns
    if column not in ["emotion", "actor_id"]
]

X = df[feature_columns].apply(
    pd.to_numeric, errors="raise"
).to_numpy(dtype=np.float32)

if not np.isfinite(X).all():
    raise ValueError(
        "Features contain NaN or infinite values. "
        "Clean the feature dataset before training."
    )

encoder = LabelEncoder()
y = encoder.fit_transform(df["emotion"].astype(str))
groups = df["actor_id"].to_numpy()

print("Dataset shape:", X.shape)
print("Emotion classes:", list(encoder.classes_))
print("Class counts:")
print(df["emotion"].value_counts())

# --------------------------------------------------
# 3. Actor-independent train/test split
# --------------------------------------------------

outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

dev_idx, test_idx = next(
    outer_split.split(X, y, groups)
)

X_dev, X_test = X[dev_idx], X[test_idx]
y_dev, y_test = y[dev_idx], y[test_idx]
groups_dev = groups[dev_idx]

dev_actors = set(groups_dev)
test_actors = set(groups[test_idx])

assert dev_actors.isdisjoint(test_actors), (
    "Actor leakage detected between development and test sets."
)

# Separate validation actors for early stopping
inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=24
)

train_rel, val_rel = next(
    inner_split.split(X_dev, y_dev, groups_dev)
)

X_train, X_val = X_dev[train_rel], X_dev[val_rel]
y_train, y_val = y_dev[train_rel], y_dev[val_rel]

print("Training samples:", len(y_train))
print("Validation samples:", len(y_val))
print("Test samples:", len(y_test))
print("Test actors:", sorted(test_actors))

# --------------------------------------------------
# 4. Scale features using training data only
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# Compensate for class imbalance in the training set
class_ids = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(class_ids, weights)
}

# --------------------------------------------------
# 5. Build the Artificial Neural Network
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(
        len(encoder.classes_),
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# 6. Train with early stopping
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=0.00001,
        verbose=1
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# --------------------------------------------------
# 7. Evaluate on untouched test actors
# --------------------------------------------------

probabilities = model.predict(X_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

print("\n========== ANN TEST RESULTS ==========")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_acc:.4f}")
print(f"Macro F1-score:    {macro_f1:.4f}")

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
)

print("\nClassification Report:\n")
print(report)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(encoder.classes_))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

display.plot(cmap="Blues", values_format="d")
plt.title("ANN - Actor-Independent Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ann_confusion_matrix.png",
    dpi=150
)
plt.close()

# Training history
plt.figure()
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ann_training_history.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 8. Save model and preprocessing artifacts
# --------------------------------------------------

model.save(MODEL_DIR / "speech_emotion_ann.keras")

joblib.dump(
    scaler,
    MODEL_DIR / "ann_scaler.joblib"
)

joblib.dump(
    encoder,
    MODEL_DIR / "ann_label_encoder.joblib"
)

joblib.dump(
    feature_columns,
    MODEL_DIR / "ann_feature_columns.joblib"
)

results = {
    "test_accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_acc),
    "macro_f1": float(macro_f1),
    "test_samples": int(len(y_test)),
    "test_actors": sorted(int(a) for a in test_actors),
    "emotion_classes": list(encoder.classes_),
    "classification_report": classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )
}

with open(
    OUTPUT_DIR / "ann_evaluation_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

print("\nANN training and evaluation completed.")
print("Model saved in:", MODEL_DIR)
print("Evaluation saved in:", OUTPUT_DIR)

