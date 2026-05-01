# FakeReview Buster

## College Mini Project

**Subject:** DMT - Data Mining Techniques  
**Project Title:** FakeReview Buster - Fake Review Detection System  
**Department:** Artificial Intelligence and Data Science  
**Institution:** K.S. Rangasamy College of Technology

FakeReview Buster is a college mini project built for the DMT subject. It is a Flask-based web application that analyzes product or service reviews and predicts whether a review is genuine, suspicious, or fake.

The project uses machine learning and data mining concepts such as text preprocessing, TF-IDF vectorization, character n-grams, heuristic feature extraction, classification, model evaluation, and pattern-based detection.

## Features

- Single review analysis
- Batch review analysis
- CSV upload support
- Fake, suspicious, and genuine verdicts
- Fake and genuine probability scores
- Signal-based explanation for detected review patterns
- Model dashboard with accuracy information
- Detection support for generic AI-generated review patterns

## Technologies Used

- Python
- Flask
- HTML, CSS, JavaScript
- Pandas
- NumPy
- Scikit-learn
- SciPy

## Machine Learning Approach

The detector combines:

- Word-level TF-IDF features
- Character n-gram TF-IDF features
- Heuristic review signals
- Logistic Regression
- Random Forest
- Gradient Boosting
- Linear SVM
- Soft Voting Classifier ensemble

The heuristic signals include review length, superlative words, generic marketing phrases, repetition, capitalization, exclamation usage, specificity, AI-style phrasing, and informal human-writing patterns.

## DMT Concepts Used

| Unit | DMT Concept |
| --- | --- |
| Unit 1 | Data objects, attributes, statistical description |
| Unit 2 | Data preprocessing, cleaning, transformation |
| Unit 3 | Pattern mining from review text |
| Unit 4 | Classification using machine learning models |
| Unit 5 | Fake/spam pattern detection and data mining trends |

## Project Structure

```text
fakereview/
├── app.py
├── detector.py
├── dataset.csv
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask app:

```bash
python app.py
```

Open the app:

```text
http://127.0.0.1:5000
```

## Sample Reviews

Fake-style review:

```text
Absolutely the best purchase I have made this decade. Cannot recommend highly enough!
```

Genuine-style review:

```text
Battery life is around 6 hours. Screen has minor glare outdoors but indoor use is fine.
```

## Team

- Aariyan O
- Jack Christopher A

## Note

This project is created for academic learning and demonstration purposes as part of the DMT mini project.
