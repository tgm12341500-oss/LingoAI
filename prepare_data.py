
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Language Detection.csv"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_PATH = OUTPUT_DIR / "cleaned_language_data.csv"

# Load original dataset
df = pd.read_csv(DATA_PATH)

# Remove empty rows
df = df.dropna(subset=["Text", "Language"])

# Remove duplicate rows
df = df.drop_duplicates()

# Remove extra spaces
df["Text"] = df["Text"].astype(str).str.strip()
df["Language"] = df["Language"].astype(str).str.strip()

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

# Display report
print("Dataset preparation completed!")
print("Original dataset: 10,337 rows")
print("Cleaned dataset:", len(df))
print("Removed duplicates:", 10337 - len(df))
print("Number of languages:", df["Language"].nunique())
print("\nLanguage distribution:")
print(df["Language"].value_counts())
print("\nSaved to:", OUTPUT_PATH)
