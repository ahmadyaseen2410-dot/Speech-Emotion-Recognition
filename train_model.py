
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# --------------------------------------------------
# 1. Paths and reproducibility
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load enhanced features
# --------------------------------------------------

csv_path = OUTPUT_DIR / "emotion_features.csv"

if not csv_path.exists():
    raise FileNotFoundError(
        f"Feature CSV not found: {csv_path}\n"
        "Run extractfeatures.py first."
    )

df = pd.read_csv(csv_path)

required_columns = {"emotion", "actor_id"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        "CSV must contain emotion and actor_id columns."
    )

feature_columns = [
    col for col in df.columns
    if col not in ["emotion", "actor_id"]
]

if not feature_columns:
    raise ValueError("No feature columns found.")

X = (
    df[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .to_numpy(dtype=np.float64)
)

y_text = df["emotion"].astype(str).to_numpy()
groups = df["actor_id"].to_numpy()

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

print("Dataset shape:", X.shape)
print("Features:", len(feature_columns))
print("Classes:", list(encoder.classes_))
print("Unique actors:", len(np.unique(groups)))

# --------------------------------------------------
# 3. Split actors into train/validation and test sets
# --------------------------------------------------
# Test actors are held out until final evaluation.

outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)

development_idx, test_idx = next(
    outer_split.split(X, y, groups=groups)
)

X_dev, X_test = X[development_idx], X[test_idx]
y_dev, y_test = y[development_idx], y[test_idx]
groups_dev = groups[development_idx]
groups_test = groups[test_idx]

assert set(groups_dev).isdisjoint(set(groups_test))

print("\nDevelopment samples:", len(development_idx))
print("Test samples:", len(test_idx))
print("Development actors:", len(np.unique(groups_dev)))
print("Test actors:", len(np.unique(groups_test)))

# --------------------------------------------------
# 4. Create actor-independent validation folds
# --------------------------------------------------
# GridSearchCV uses only development actors.
# Every CV validation fold holds out entire actors.

inner_split = GroupShuffleSplit(
    n_splits=4,
    test_size=0.25,
    random_state=RANDOM_STATE
)

cv_splits = list(
    inner_split.split(X_dev, y_dev, groups=groups_dev)
)

# --------------------------------------------------
# 5. Build SVM pipeline and tune hyperparameters
# --------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        class_weight="balanced"
    ))
])

parameter_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1],
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="f1_macro",
    cv=cv_splits,
    n_jobs=-1,
    refit=True,
    verbose=1,
    error_score="raise"
)

print("\nTuning SVM parameters...")
search.fit(X_dev, y_dev)

model = search.best_estimator_

print("\nBest parameters:", search.best_params_)
print("Best cross-validation macro F1:",
      round(search.best_score_, 4))

# --------------------------------------------------
# 6. Final evaluation on unseen actors
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

print("\nFinal test accuracy:", round(accuracy, 4))
print(f"Final test accuracy percentage: {accuracy * 100:.2f}%")
print("Final test macro F1:", round(macro_f1, 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

# --------------------------------------------------
# 7. Save confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_))
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot(cmap="Blues", values_format="d")
plt.title("RBF-SVM Speech Emotion Recognition")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "svm_confusion_matrix.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 8. Save model and preprocessing information
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_DIR / "speech_emotion_svm.joblib"
)

joblib.dump(
    encoder,
    MODEL_DIR / "label_encoder.joblib"
)

joblib.dump(
    feature_columns,
    MODEL_DIR / "feature_columns.joblib"
)

results = {
    "test_accuracy": accuracy,
    "test_macro_f1": macro_f1,
    "best_cv_macro_f1": search.best_score_,
    "best_parameters": search.best_params_,
    "test_actors": sorted(
        int(actor) for actor in np.unique(groups_test)
    ),
}

joblib.dump(
    results,
    OUTPUT_DIR / "svm_evaluation_results.joblib"
)

print("\nTraining and evaluation completed.")
print("SVM model saved to:", MODEL_DIR / "speech_emotion_svm.joblib")
print("Confusion matrix saved to:", OUTPUT_DIR / "svm_confusion_matrix.png")
print("Feature order saved to:", MODEL_DIR / "feature_columns.joblib")
