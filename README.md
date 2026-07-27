
# Educational Process Automation: Improving Student Engagement Using AI

## Overview
This project uses machine learning models to analyze student data and predict exam performance. It aims to help educators improve engagement through data-driven insights.

## Dataset
The dataset (`student_data_cleaned.csv`) includes:
- Hours_Studied
- Attendance
- Sleep_Hours
- Previous_Scores
- Exam_Score

## Tools Used
- Python (Pandas, Scikit-learn, Matplotlib)
- Streamlit for dashboard visualization
- GitHub for version control

## How to Run
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app:  
   ```bash
   streamlit run app.py

Results
Linear Regression achieved better accuracy (R² = 0.57, MSE = 6.59) compared to Random Forest.
1. Hours Studied vs Exam Score 
The scatter plot below illustrates the relationship between study hours and exam performance.  
It shows that students who studied more hours generally achieved higher exam scores, though there is some variation due to other factors such as sleep and attendance.
<img width="1022" height="782" alt="exam_score_trends jpeg" src="https://github.com/user-attachments/assets/0bcd623f-0202-44c1-9b77-18431df86bdb" />


Future Work
Enhance the model with more features and deploy it for real-time student monitoring.

Author
Pragati Kamat – MCA Student, Amrita Vishwa Vidyapeetham


3. Attendance vs Exam Score
This scatter plot shows a positive correlation between attendance and exam performance. Students with higher attendance generally score better, highlighting the importance of consistent participation.
<img width="950" height="737" alt="attendance_vs_exam_score jpeg" src="https://github.com/user-attachments/assets/4ca202f2-db07-4fc0-a2e0-0e978d703ae4" />

