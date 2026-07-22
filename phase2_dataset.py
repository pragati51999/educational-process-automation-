import pandas as pd

# Load CSV dataset
data = pd.read_csv("data/StudentPerformanceFactors.csv")

# Feature selection (optional)
data = data[['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores','Exam_Score']]

# Clean data
data = data.dropna().drop_duplicates()
data = pd.get_dummies(data, drop_first=True)

# Save cleaned dataset
data.to_csv("data/student_data_cleaned.csv", index=False)
print("Cleaned dataset saved successfully!")
