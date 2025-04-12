# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
 
# Load dataset (replace with your path or use an online URL)
df = pd.read_csv('C:\\Users\\hi\\Downloads\\StudentsPerformance.csv')  # Example dataset
print(df.head())
# Check null values
print(df.isnull().sum())

# Encode categorical variables
label_enc = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_enc.fit_transform(df[col])

# Create a new target column: average score
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Define features and target
X = df.drop(['math score', 'reading score', 'writing score', 'average_score'], axis=1)
y = df['average_score']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import LabelEncoder

# Encode all categorical columns
label_enc = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = label_enc.fit_transform(X[col])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

print("R^2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

print("R^2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Average Score")
plt.ylabel("Predicted Average Score")
plt.title("Actual vs Predicted Student Scores")
plt.grid(True)
plt.show()

import joblib

joblib.dump(model, 'student_score_predictor.pkl')
joblib.dump(scaler, 'scaler.pkl')

def predict_student_score(input_dict):
    df_input = pd.DataFrame([input_dict])

    for col in df_input.columns:
        if df_input[col].dtype == 'object':
            df_input[col] = label_enc.fit_transform(df_input[col])

    input_scaled = scaler.transform(df_input)
    return model.predict(input_scaled)[0]
