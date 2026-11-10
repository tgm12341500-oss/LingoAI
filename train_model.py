
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_language_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

X = df["Text"].astype(str)
y = df["Language"].astype(str)

# Encode language labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data before fitting TF-IDF to avoid data leakage
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    max_features=15000
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train model
model = LinearSVC(C=1.0, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n========== LingoAI Evaluation ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# Save model files in the project folder
joblib.dump(model, BASE_DIR / "language_model.pkl")
joblib.dump(vectorizer, BASE_DIR / "tfidf.pkl")
joblib.dump(label_encoder, BASE_DIR / "label_encoder.pkl")

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=range(len(label_encoder.classes_))
)

fig, ax = plt.subplots(figsize=(14, 12))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    colorbar=False
)

plt.title("LingoAI - Confusion Matrix")
plt.tight_layout()

chart_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

# Save evaluation summary
report = classification_report(
    y_test,
    y_pred,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(OUTPUT_DIR / "evaluation_report.csv")

print("\nModel files saved successfully!")
print("Confusion Matrix:", chart_path)
print("Evaluation Report:", OUTPUT_DIR / "evaluation_report.csv")
print("\n=======================================")
