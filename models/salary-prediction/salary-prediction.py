# ===========================
# 1. Import libraries
# ===========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib   # for saving/loading the model

# ===========================
# 2. Load raw dataset
# ===========================
df = pd.read_csv("C:\\Users\\Med_Bk\\hr-automation\\models\\salary-prediction\\salary_data.csv")

# Drop ID column
df = df.drop(columns=["id"])

# ===========================
# 3. Cleaning functions
# ===========================
def clean_role(df):
    df['role'] = df['role'].astype(str).str.strip().str.title()
    df = pd.get_dummies(df, columns=['role'], prefix='role')
    return df

def clean_years_experience(df):
    df['years_experience'] = pd.to_numeric(df['years_experience'], errors='coerce')
    median_val = df['years_experience'].median()
    df['years_experience'] = df['years_experience'].fillna(median_val)
    return df

def clean_degree(df):
    df['degree'] = df['degree'].astype(str).str.strip().str.title()
    df = pd.get_dummies(df, columns=['degree'], prefix='degree')
    return df

def clean_company_size(df):
    df['company_size'] = df['company_size'].astype(str).str.strip().str.title()
    df = pd.get_dummies(df, columns=['company_size'], prefix='company')
    return df

def clean_location(df):
    df['location'] = df['location'].astype(str).str.strip().str.title()
    df = pd.get_dummies(df, columns=['location'], prefix='location')
    return df

def clean_level(df):
    df['level'] = df['level'].astype(str).str.strip().str.lower()
    level_map = {"intern": 0, "junior": 1, "mid": 2, "senior": 3}
    df['level'] = df['level'].map(level_map)
    if df['level'].notna().any():
        fill_value = int(df['level'].mode()[0])
    else:
        fill_value = 0
    df['level'] = df['level'].fillna(fill_value).astype(int)
    return df

def clean_salary(df):
    df['salary_mad'] = pd.to_numeric(df['salary_mad'], errors='coerce')
    median_val = df['salary_mad'].median()
    df['salary_mad'] = df['salary_mad'].fillna(median_val)
    return df

# ===========================
# 4. Apply cleaning
# ===========================
df = clean_role(df)
df = clean_years_experience(df)
df = clean_degree(df)
df = clean_company_size(df)
df = clean_location(df)
df = clean_level(df)
df = clean_salary(df)

# Convert boolean to int
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Save cleaned dataset
df.to_csv("hr_salary_cleaned.csv", index=False)
print("✅ Dataset saved as hr_salary_cleaned.csv")

# ===========================
# 5. Visualization
# ===========================
numeric_cols = ['years_experience', 'level', 'salary_mad']

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.show()

# Average salary by role
role_cols = [col for col in df.columns if col.startswith('role_')]
avg_salary_roles = {col: df[df[col] == 1]['salary_mad'].mean() for col in role_cols}

plt.figure(figsize=(12,6))
sns.barplot(x=list(avg_salary_roles.keys()), y=list(avg_salary_roles.values()))
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average Salary (MAD)")
plt.title("Average Salary by Role")
plt.show()

# ===========================
# 6. Train/Test Split
# ===========================
X = df.drop(columns=['salary_mad'])
y = df['salary_mad']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ===========================
# 7. Train Model
# ===========================
model = LinearRegression()
model.fit(X_train, y_train)

# ===========================
# 8. Evaluate Model
# ===========================
y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Comparison table
comparison = X_test.copy()
comparison['Actual_Salary'] = y_test
comparison['Predicted_Salary'] = y_pred
print(comparison.head(10))

# Scatter plot (Predicted vs Actual)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Salary (MAD)")
plt.ylabel("Predicted Salary (MAD)")
plt.title("Predicted vs Actual Salary")
plt.show()

# Bar comparison (first 20 samples)
comparison_sample = comparison.head(20)
comparison_sample[['Actual_Salary', 'Predicted_Salary']].plot(kind='bar', figsize=(12,6))
plt.title("Comparison of Actual vs Predicted Salaries")
plt.ylabel("Salary (MAD)")
plt.xticks(rotation=45)
plt.show()

# ===========================
# 9. Export Model
# ===========================
joblib.dump(model, "salary_prediction_model.pkl")
joblib.dump(X.columns.tolist(), "encoder_columns.pkl")
print("✅ Model and encoder columns exported successfully!")

# ===========================
# 10. Predict Function (with exported model)
# ===========================
def predict_salary(candidate_info):
    model = joblib.load("salary_prediction_model.pkl")
    encoder_columns = joblib.load("encoder_columns.pkl")

    df = pd.DataFrame([candidate_info])
    categorical_cols = ['role', 'degree', 'company_size', 'location', 'level']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    for col in encoder_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[encoder_columns]

    predicted_salary = model.predict(df_encoded)[0]
    return round(predicted_salary, 2)

# ===========================
# 11. Example Candidate Test
# ===========================
candidate = {
    "role": "SW Engineer",
    "years_experience": 1.5,
    "degree": "Masters",
    "company_size": "Mid",
    "location": "Casablanca",
    "level": "Junior"
}

print(f"💰 Predicted Salary: {predict_salary(candidate)} MAD")
