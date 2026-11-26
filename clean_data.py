import os
import pandas as pd

def clean_data(input_path, output_path):
    student = pd.read_csv(input_path)
    
     #check for any NaN/Null values
    null_counts = student.isnull().sum().sum()
    print(f" Found {null_counts} missing values in the dataset.")

    if null_counts > 0:
        #Used forward fill to carry the last known price across any weekend/holiday gaps
        student = student.ffill()
        print("For the missing values have been handled using forward fill (ffill).")

    duplicate_count =student.duplicated().sum()
    print(f" Found {duplicate_count} duplicate rows in the dataset.")
    if duplicate_count > 0:
        student = student.drop_duplicates()
        print(f"Found and removed {duplicate_count} duplicate rows.")
        print(student.columns)
     #now to save the clean dataset
    student.to_csv(output_path, index=False)
    print(f"Cleaned data successfully saved to {output_path}. ")
    print("\nSample of clean data:")
    print(student.head())
    print(student.describe())
    cols = [ 'gender', 'school_type', 'parent_education','internet_access','travel_time','extra_activities',
            'study_method']
    student = pd.get_dummies(student, columns = cols, drop_first=True, dtype = int, sparse = False)

    
if __name__ == "__main__":
    # collecting the raw dataset from the data folder
    RAW_PATH = os.path.join("data", "Student_Performance.csv")
    # creating the clean dataset and added to the data folder
    CLEAN_PATH = os.path.join("data", "clean_student_performance.csv")

    clean_data(RAW_PATH, CLEAN_PATH)