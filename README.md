
# LingoAI – AI Language Detection System

LingoAI is a desktop application that uses Machine Learning and Natural Language Processing (NLP) to automatically identify the language of a given text.

The system is built with Python and provides an interactive graphical interface for language detection, prediction history, and model evaluation.

## Features

- Detects text in 17 different languages.
- Machine Learning-based language classification.
- TF-IDF character n-gram feature extraction.
- Linear Support Vector Machine (Linear SVM).
- Interactive desktop interface built with CustomTkinter.
- Real-time word and character counter.
- Detection history saved locally.
- Export detection history to CSV.
- Model evaluation using accuracy, precision, recall, and F1-score.
- Confusion matrix visualization.

## Supported Languages

Arabic, Danish, Dutch, English, French, German, Greek, Hindi, Italian, Kannada, Malayalam, Portuguese, Russian, Spanish, Swedish, Tamil, and Turkish.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- TF-IDF
- Linear Support Vector Machine (LinearSVC)
- CustomTkinter
- Matplotlib
- Joblib

## Model Performance

The model was evaluated using a stratified train-test split, with 80% of the data used for training and 20% for testing.

| Metric | Result |
|---|---:|
| Accuracy | 98.83% |
| Test Samples | 2,055 |
| Supported Languages | 17 |

The evaluation also includes precision, recall, F1-score, and a confusion matrix.

**Note:** These results represent performance on the selected test split and do not guarantee the same accuracy on all real-world text.

## Project Structure

```text
LanguageApp/
│
├── app.py
├── app_backup.py
├── train_model.py
├── prepare_data.py
├── language_detection.ipynb
│
├── language_model.pkl
├── tfidf.pkl
├── label_encoder.pkl
│
├── data/
│   ├── Language Detection.csv
│   └── cleaned_language_data.csv
│
├── outputs/
│   ├── confusion_matrix.png
│   └── evaluation_report.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project directory:

```bash
cd LanguageApp
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

## Train the Model

To prepare the dataset:

```bash
python prepare_data.py
```

To train and evaluate the model:

```bash
python train_model.py
```

The trained model files and evaluation outputs will be saved in the project directory.

## Dataset

The project uses a language detection dataset containing text samples labeled with their corresponding languages.

The dataset is used for training, testing, and evaluating the language classification model.

## Author

**Tasneem Gamal Mohamed**

Computer Science Graduate

---

*LingoAI – Intelligent Language Detection Using Machine Learning.*
