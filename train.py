import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(DATA_PATH, model_path):
    student = pd.read_csv(DATA_PATH) 
    
    # 1. Separate X and y - y stays as ONE column
    X = student.drop(columns=['final_grade','overall_score',"travel_time"])
    y = student["final_grade"]

    # 2. One-hot encode ONLY the features (this fixes the 'male' error)
    X = pd.get_dummies(X)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Scale numeric columns
    numeric_cols = X_train.select_dtypes(include=["int64","float64"]).columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    scaler = StandardScaler()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 5. Train model
    model = RandomForestClassifier(n_estimators=300, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    prediction = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, prediction)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, prediction))
    
    # 6. Save the trained model for the backend
    joblib.dump(model, model_path)
    print(f"\nSuccessful! Model saved to {model_path}. It is now ready for the API")
    joblib.dump(list(X_train_scaled.columns.tolist()), "app/model_columns.pkl")
    joblib.dump(scaler, "app/scaler.pkl")
    joblib.dump(numeric_cols, "app/feature_columns.pkl")  # Save the feature columns for later use in the API
if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "clean_student_performance.csv")
    os.makedirs("app", exist_ok=True)
    MODEL_PATH = os.path.join("app", "student_performance_model.pkl")
    FEATURE_COLUMNS_PATH = os.path.join("app", "feature_columns.pkl")
    
    train_model(DATA_PATH, MODEL_PATH)