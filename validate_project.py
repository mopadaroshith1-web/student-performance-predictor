
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
file_path = Path(__file__).parent / "student_data.csv"

if not file_path.exists():
    raise FileNotFoundError(
        "student_data.csv not found. Run your Streamlit app first."
    )

df = pd.read_csv(file_path)

features = [
    "Study Hours",
    "Attendance",
    "Previous Score",
    "Assignments Score"
]
target = "Final Marks"

# 2. Check required columns
required_columns = features + [target]
missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# 3. Convert columns to numeric values
for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows without a target value
df = df.dropna(subset=[target])

# Keep only values within valid ranges
valid_rows = (
    df["Study Hours"].between(0, 24)
    & df["Attendance"].between(0, 100)
    & df["Previous Score"].between(0, 100)
    & df["Assignments Score"].between(0, 100)
    & df[target].between(0, 100)
)

df = df.loc[valid_rows].copy()

if len(df) < 10:
    raise ValueError("At least 10 valid rows are required.")

print("DATASET VALIDATION")
print("------------------")
print("Valid rows:", len(df))
print("Duplicate rows:", df.duplicated().sum())
print("Missing feature values:", df[features].isna().sum().sum())
print("Invalid final marks:", (~df[target].between(0, 100)).sum())

# 4. Split the data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. Define machine learning models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42
    )
}

# 6. Train and evaluate each model
print("\nMODEL VALIDATION")
print("----------------")

for name, estimator in models.items():
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        estimator
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert np.isfinite(predictions).all(), (
        f"{name} returned invalid predictions."
    )

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"MAE:  {mae:.3f}")
    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

print("\nSUCCESS: All three models trained and produced valid predictions.")
