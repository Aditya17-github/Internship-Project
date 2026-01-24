# ==========================
# EMPLOYEE ATTRITION PROJECT
# ==========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fix matplotlib display
plt.tight_layout()
sns.set(style="whitegrid")

# Load dataset (IMPORTANT: tab separated)
df = pd.read_csv("HR_Employee_Attrition.csv", sep="\t")

print("Dataset Loaded Successfully!\n")

# ==========================
# 1. BASIC INFO & OVERVIEW
# ==========================
print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())
print("\nTop 5 Rows:\n", df.head())
print("\nLast 5 Rows:\n", df.tail())

print("\nColumn Data Types:\n", df.dtypes)

# ==========================
# 2. UNIQUE VALUES OF EACH COLUMN
# ==========================
print("\n====== UNIQUE VALUES IN COLUMNS ======")
for col in df.columns:
    print(f"\n{col}: {df[col].unique()[:10]}")  # show first 10 unique values only

# ==========================
# 3. ATTRITION DISTRIBUTION (PIE)
# ==========================
attrition_counts = df['Attrition'].value_counts()

plt.figure(figsize=(6,6))
attrition_counts.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title("Attrition Distribution")
plt.ylabel("")
plt.show()

# ==========================
# 4. AGE DISTRIBUTION (HISTOGRAM)
# ==========================
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Age', hue='Attrition', kde=True, bins=20)
plt.title("Age Distribution by Attrition")
plt.show()

# ==========================
# 5. JOB SATISFACTION (BAR + PIE)
# ==========================
plt.figure(figsize=(6,4))
sns.countplot(x='JobSatisfaction', data=df)
plt.title("Job Satisfaction Count")
plt.show()

# Pie chart for Job Satisfaction
job_sat = df['JobSatisfaction'].value_counts()
plt.figure(figsize=(6,6))
job_sat.plot.pie(autopct='%1.1f%%')
plt.title("Job Satisfaction Distribution")
plt.ylabel("")
plt.show()

# ==========================
# 6. DEPARTMENT DISTRIBUTION
# ==========================
plt.figure(figsize=(7,5))
sns.countplot(x='Department', data=df)
plt.title("Department Distribution")
plt.xticks(rotation=20)
plt.show()

# ==========================
# 7. EDUCATION FIELD
# ==========================
plt.figure(figsize=(8,5))
sns.countplot(x='EducationField', data=df)
plt.title("Education Field Distribution")
plt.xticks(rotation=20)
plt.show()

# ==========================
# 8. WORK-LIFE BALANCE
# ==========================
plt.figure(figsize=(6,4))
sns.countplot(x='WorkLifeBalance', data=df)
plt.title("Work-Life Balance Count")
plt.show()

# Pie chart Worklife Balance
wlb = df['WorkLifeBalance'].value_counts()
plt.figure(figsize=(6,6))
wlb.plot.pie(autopct='%1.1f%%')
plt.title("Work-Life Balance Pie")
plt.ylabel("")
plt.show()

print("\nAnalysis Completed Successfully!")
