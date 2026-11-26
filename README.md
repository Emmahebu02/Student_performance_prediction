
# Student Performance Predictor

A machine learning system that predicts a student's final grade (A–F) from demographics, study habits, attendance, and previous academic performance. The model is served as a REST API using FastAPI.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [API Usage](#api-usage)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

## Overview

This project covers the full ML workflow:

1. **Data Cleaning** — handle missing values, duplicates, and inconsistent categories
2. **Preprocessing** — one-hot encoding for categorical features, StandardScaler for numeric features
3. **Training** — Random Forest Classifier with balanced class weights
4. **Deployment** — FastAPI endpoint for real-time predictions

The goal is not just high accuracy, but a reproducible pipeline from raw CSV to deployed API.

## Project Structure

student-performance-predictor/
├── app/
│   ├── main.py                       # FastAPI application
│   ├── student_performance_model.pkl # Trained model
│   ├── model_columns.pkl             # Feature columns used in training
│   └── scaler.pkl                    # Fitted StandardScaler
├── data/
│   ├── student_performance.csv       # Raw dataset
│   └── clean_student_performance.csv # Cleaned dataset
├── clean_data.py                     # Data cleaning pipeline
├── train.py                          # Model training pipeline
├── requirements.txt
└── README.md

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/YOUR-USERNAME/student-performance-predictor.git
cd student-performance-predictor

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt
## Training

Run the pipeline in order:
# 1. Clean raw data
python clean_data.py

# 2. Train and save model + scaler + columns
python train.py
Expected output:
Accuracy: 0.76
Successful! Model saved to app/student_performance_model.pkl
## API Usage

Start the API server:
uvicorn app.main:app --reload
API docs (Swagger UI): *http://127.0.0.1:8000/docs*

### Predict Endpoint

*POST* `/predict`

Request body:
{
  age: int
    gender: Literal["female", "male", "other"]
    school_type: Literal["private", "public"]
    parent_education: Literal["no formal", "high school", "diploma", "graduate", "post graduate", "phd"]
    study_hours: float
    attendance_percentage: float
    internet_access: Literal["yes", "no"]
    extra_activities: Literal["yes", "no"]
    study_method: Literal["mixed", "notes", "textbook", "group study", "coaching", "online videos"]
    math_score: float
    science_score: float
    english_score: float
}
Response:
{
  "predicted_final_grade": "b"
}
> Note: Request fields must match the columns in `clean_student_performance.csv` (excluding `final_grade`).

## Model Performance

*Model:* Random Forest Classifier  
`n_estimators=300, class_weight='balanced'`
Metric | Score
Accuracy | 0.76
Macro Avg F1 | 0.72
Weighted Avg F1 | 0.76
Detailed classification report:
Grade | Precision | Recall | F1-Score | Support
a | 0.68 | 0.78 | 0.73 | 60
b | 0.83 | 0.90 | 0.86 | 156
c | 0.76 | 0.74 | 0.75 | 138
d | 0.57 | 0.70 | 0.63 | 47
e | 0.70 | 0.43 | 0.53 | 30
f | 0.80 | 0.82 | 0.81 | 69
*Observation:* The model performs best on grades B, C, and F. Performance on grade E is lower due to limited samples and class imbalance, which is partly mitigated with `class_weight='balanced'`.

## Tech Stack

- *Python* — core language
- *Pandas, NumPy* — data manipulation
- *Scikit-learn* — preprocessing and modeling
- *FastAPI, Uvicorn* — API serving
- *Joblib* — model persistence

## Future Improvements

- Compare Random Forest with XGBoost and Logistic Regression
- Add cross-validation and hyperparameter tuning
- Build a simple frontend with Streamlit
- Add tests for API endpoints
- Deploy API to Render or Railway

## Author

Built as a learning project covering data cleaning, model training, and API deployment.

