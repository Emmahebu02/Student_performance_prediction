from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Student Performance API")

model = joblib.load("app/student_performance_model.pkl")
model_columns = joblib.load("app/model_columns.pkl")
scaler = joblib.load("app/scaler.pkl")
numeric_cols = list(joblib.load("app/feature_columns.pkl"))

class StudentInput(BaseModel):
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

@app.get("/")
def home():
    return {"message": "Student Performance API is running"}

@app.post("/predict")
def predict(student: StudentInput):
    df = pd.DataFrame([student.model_dump()])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)
    df[numeric_cols] = df[numeric_cols].astype(float)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return {"predicted_final_grade": str(model.predict(df)[0])}