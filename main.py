import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_students = 200
data = {
    'GPA_Semester1': np.random.uniform(0, 4, num_students),
    'GPA_Semester2': np.random.uniform(0, 4, num_students),
    'Attendance_Semester1': np.random.randint(0, 101, num_students), # Percentage
    'Attendance_Semester2': np.random.randint(0, 101, num_students),
    'AtRisk': np.random.choice([0, 1], num_students, p=[0.8, 0.2]) # 20% at risk
}
df = pd.DataFrame(data)
# Add some noise to make it more realistic
df['GPA_Semester2'] += np.random.normal(0, 0.2, num_students)
df['Attendance_Semester2'] += np.random.normal(0, 5, num_students)
df['GPA_Semester2'] = df['GPA_Semester2'].clip(0,4) #Ensure GPA stays within bounds
df['Attendance_Semester2'] = df['Attendance_Semester2'].clip(0,100) #Ensure attendance stays within bounds
# --- 2. Data Cleaning and Feature Engineering ---
#Handle potential outliers (example - extreme low attendance)
df['Attendance_Semester1'] = df['Attendance_Semester1'].apply(lambda x: 0 if x < 10 else x)
df['Attendance_Semester2'] = df['Attendance_Semester2'].apply(lambda x: 0 if x < 10 else x)
# Create a combined GPA
df['GPA_Combined'] = (df['GPA_Semester1'] + df['GPA_Semester2']) / 2
# Create a combined attendance
df['Attendance_Combined'] = (df['Attendance_Semester1'] + df['Attendance_Semester2']) / 2
# --- 3. Predictive Modeling ---
X = df[['GPA_Combined', 'Attendance_Combined']]
y = df['AtRisk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# --- 4. Model Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
plt.scatter(df['GPA_Combined'], df['Attendance_Combined'], c=df['AtRisk'], cmap='viridis')
plt.xlabel('Combined GPA')
plt.ylabel('Combined Attendance')
plt.title('Student At-Risk Status')
plt.colorbar(label='At Risk (1=Yes, 0=No)')
plt.savefig('student_risk_scatter.png')
print("Plot saved to student_risk_scatter.png")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not At Risk', 'At Risk'], yticklabels=['Not At Risk', 'At Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Plot saved to confusion_matrix.png")