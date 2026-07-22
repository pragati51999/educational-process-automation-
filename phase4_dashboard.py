import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
data = pd.read_csv("data/student_data_cleaned.csv")

# Load models
lr_model = joblib.load("models/linear_regression_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")

st.title("Student Performance Dashboard")

# Show dataset preview
st.subheader("Dataset Preview")
st.write(data.head())

# Input form for prediction
st.subheader("Predict Exam Score")
hours = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
sleep = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
previous = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)

input_data = pd.DataFrame([[hours, attendance, sleep, previous]],
                          columns=['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores'])

# Predictions
lr_pred = lr_model.predict(input_data)[0]
rf_pred = rf_model.predict(input_data)[0]

st.write(f"Linear Regression Prediction: {lr_pred:.2f}")
st.write(f"Random Forest Prediction: {rf_pred:.2f}")




st.info("Note: Increasing study hours, attendance, and sleep quality generally improves predicted scores.")


# Visualization
st.subheader("Exam Score Trends")
fig, ax = plt.subplots()
sns.scatterplot(x=data['Hours_Studied'], y=data['Exam_Score'], ax=ax)
ax.set_title("Hours Studied vs Exam Score")
st.pyplot(fig)
# Second visualization
st.subheader("Attendance vs Exam Score")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=data['Attendance'], y=data['Exam_Score'], ax=ax2)
ax2.set_title("Attendance vs Exam Score")
st.pyplot(fig2)
# At-Risk Students Table
st.subheader("At-Risk Students (Predicted < 40)")
data['Predicted_Score'] = lr_model.predict(
    data[['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']]
)
at_risk = data[data['Predicted_Score'] < 40]

if at_risk.empty:
    st.success("No students predicted to score below 40 ✅")
else:
    st.warning("Students predicted to score below 40 ⚠️")
    st.write(at_risk)
# Performance Summary
st.subheader("Performance Summary")
avg_score = data['Exam_Score'].mean()
col1, col2, col3 = st.columns(3)
col1.metric("Average Score", f"{avg_score:.2f}")
col2.metric("Highest Score", data['Exam_Score'].max())
col3.metric("Lowest Score", data['Exam_Score'].min())
st.caption("Class performance overview based on current dataset.")
# Dynamic color progress bar
if avg_score > 60:
    bar_color = "green"
elif avg_score >= 40:
    bar_color = "yellow"
else:
    bar_color = "red"

st.markdown(
    f"""
    <div style="background-color:lightgray;border-radius:10px;">
        <div style="width:{avg_score}%;background-color:{bar_color};
        height:20px;border-radius:10px;"></div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Average score progress out of 100")


# Histogram of student scores (score ranges)
import matplotlib.pyplot as plt
st.subheader("Score Range Distribution")
fig, ax = plt.subplots()
ax.hist(data['Exam_Score'], bins=[0, 40, 60, 80, 100],
        color='skyblue', edgecolor='black')
ax.set_xlabel("Score Range")
ax.set_ylabel("Number of Students")
st.pyplot(fig)

# -------------------------------
# Chatbot Section
# -------------------------------


st.subheader("Student Support Chatbot")

def chatbot_response(user_input):
    if "score now" in user_input.lower() or "current score" in user_input.lower():
        return "Your current score is 72 based on the latest exam."
    elif "improve my score" in user_input.lower() or "score" in user_input.lower():
        return "To improve your score, study regularly, practice past papers, and review mistakes after each test."
    elif "improve attendance" in user_input.lower() or "attendance" in user_input.lower():
        return "To improve attendance, set a routine, avoid unnecessary absences, and remember that higher attendance often leads to better scores."
    elif "progress" in user_input.lower():
        return "Your average score is 72. You improved by 10% compared to last test."
    
    elif "resources" in user_input.lower():
        return "I recommend Khan Academy and Coursera for extra practice."
    elif "exam" in user_input.lower():
        return "Your next exam is scheduled for 15th July. Prepare early."
    else:
        return "Sorry, I can only answer questions about performance, attendance, and resources."

user_query = st.text_input("Ask me a question:")
if user_query:
    st.write("**You:**", user_query)
    st.write("**Chatbot:**", chatbot_response(user_query))
