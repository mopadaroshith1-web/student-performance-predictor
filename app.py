
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Analyze student data, compare machine learning models, "
    "and predict final examination marks."
)

DATA_FILE = Path(__file__).parent / "student_data.csv"

FEATURES = [
    "Study Hours",
    "Attendance",
    "Previous Score",
    "Assignments Score"
]
TARGET = "Final Marks"


# --------------------------------------------------
# 2. CREATE A SYNTHETIC DATASET
# --------------------------------------------------

def create_dataset(number_of_students=500):
    rng = np.random.default_rng(42)

    study_hours = rng.uniform(0.5, 10, number_of_students)
    attendance = rng.uniform(40, 100, number_of_students)
    previous_score = rng.uniform(20, 100, number_of_students)
    assignment_score = rng.uniform(20, 100, number_of_students)

    # Artificial relationship used to generate example marks.
    noise = rng.normal(0, 5, number_of_students)

    final_marks = (
        0.9 * study_hours
        + 0.15 * attendance
        + 0.35 * previous_score
        + 0.30 * assignment_score
        + noise
        + 3
    )

    final_marks = np.clip(final_marks, 0, 100)

    return pd.DataFrame({
        "Study Hours": np.round(study_hours, 2),
        "Attendance": np.round(attendance, 2),
        "Previous Score": np.round(previous_score, 2),
        "Assignments Score": np.round(assignment_score, 2),
        "Final Marks": np.round(final_marks, 2)
    })


# Create the CSV file if it does not exist.
if not DATA_FILE.exists():
    create_dataset().to_csv(DATA_FILE, index=False)


# --------------------------------------------------
# 3. LOAD AND PREPROCESS THE DATA
# --------------------------------------------------

@st.cache_data
def load_data(file_path, modified_time):
    data = pd.read_csv(file_path)

    for column in FEATURES + [TARGET]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    # Remove rows where the target is missing.
    data = data.dropna(subset=[TARGET])

    # Keep plausible values within the expected ranges.
    data = data[
        data["Study Hours"].between(0, 24)
        & data["Attendance"].between(0, 100)
        & data["Previous Score"].between(0, 100)
        & data["Assignments Score"].between(0, 100)
        & data[TARGET].between(0, 100)
    ].copy()

    return data


data = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)

if len(data) < 10:
    st.error(
        "The dataset needs at least 10 valid records. "
        "Check student_data.csv."
    )
    st.stop()

st.sidebar.header("Dataset")
st.sidebar.write(f"Valid student records: {len(data)}")
st.sidebar.write("Dataset file: student_data.csv")

if st.sidebar.button("Regenerate example dataset"):
    create_dataset().to_csv(DATA_FILE, index=False)
    st.cache_data.clear()
    st.rerun()


# --------------------------------------------------
# 4. EXPLORE THE DATA
# --------------------------------------------------

st.header("1. Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Students", len(data))
c2.metric("Average Final Marks", f"{data[TARGET].mean():.2f}")
c3.metric("Highest Final Marks", f"{data[TARGET].max():.2f}")

with st.expander("View student dataset"):
    st.dataframe(data, use_container_width=True)

with st.expander("View descriptive statistics"):
    st.dataframe(data.describe().round(2), use_container_width=True)


# --------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

st.header("2. Exploratory Data Analysis")

left, right = st.columns(2)

with left:
    st.subheader("Distribution of Final Marks")
    fig, ax = plt.subplots()
    ax.hist(data[TARGET], bins=20, edgecolor="black")
    ax.set_xlabel("Final Marks")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Student Marks")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Study Hours vs Final Marks")
    fig, ax = plt.subplots()
    ax.scatter(data["Study Hours"], data[TARGET], alpha=0.6)
    ax.set_xlabel("Study Hours per Day")
    ax.set_ylabel("Final Marks")
    ax.set_title("Study Hours and Performance")
    st.pyplot(fig)
    plt.close(fig)

left, right = st.columns(2)

with left:
    st.subheader("Attendance vs Final Marks")
    fig, ax = plt.subplots()
    ax.scatter(data["Attendance"], data[TARGET], alpha=0.6)
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Final Marks")
    ax.set_title("Attendance and Performance")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Feature Correlation")
    fig, ax = plt.subplots()
    sns.heatmap(
        data[FEATURES + [TARGET]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 6. SPLIT THE DATA
# --------------------------------------------------

X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 7. BUILD AND TRAIN THREE MODELS
# --------------------------------------------------

models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(
            max_depth=5,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        ))
    ])
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    predictions[name] = predicted

    mae = mean_absolute_error(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predicted)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R² Score": r2
    })

results_df = pd.DataFrame(results).sort_values("MAE")


# --------------------------------------------------
# 8. EVALUATE AND COMPARE MODELS
# --------------------------------------------------

st.header("3. Model Training and Evaluation")

st.write(
    f"Training records: {len(X_train)} | "
    f"Testing records: {len(X_test)}"
)

st.caption(
    "The test records are kept separate from training. "
    "Lower MAE and MSE generally indicate smaller prediction errors; "
    "R² closer to 1 indicates a better fit."
)

st.dataframe(
    results_df.round(3),
    use_container_width=True,
    hide_index=True
)

best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]

st.success(f"Lowest test MAE in this comparison: {best_name}")

st.subheader("Actual vs Predicted Marks")

comparison = pd.DataFrame({
    "Actual Marks": y_test.to_numpy(),
    "Predicted Marks": predictions[best_name]
})

fig, ax = plt.subplots()
ax.scatter(
    comparison["Actual Marks"],
    comparison["Predicted Marks"],
    alpha=0.7
)
ax.plot([0, 100], [0, 100], linestyle="--")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("Actual Marks")
ax.set_ylabel("Predicted Marks")
ax.set_title(f"Actual vs Predicted: {best_name}")
st.pyplot(fig)
plt.close(fig)

with st.expander("View test predictions"):
    st.dataframe(comparison.round(2), use_container_width=True)

st.download_button(
    "Download test predictions as CSV",
    comparison.to_csv(index=False).encode("utf-8"),
    file_name="test_predictions.csv",
    mime="text/csv"
)


# --------------------------------------------------
# 9. PREDICT A NEW STUDENT'S MARKS
# --------------------------------------------------

st.header("4. Predict a Student's Performance")

st.write(
    "Enter a student's academic information to estimate "
    "their final marks using the selected model."
)

with st.form("student_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        new_study_hours = st.number_input(
            "Study Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5
        )

        new_attendance = st.slider(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=75
        )

    with col2:
        new_previous_score = st.number_input(
            "Previous Score",
            min_value=0.0,
            max_value=100.0,
            value=65.0
        )

        new_assignment_score = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

    submitted = st.form_submit_button("Predict Final Marks")

if submitted:
    new_student = pd.DataFrame([{
        "Study Hours": new_study_hours,
        "Attendance": new_attendance,
        "Previous Score": new_previous_score,
        "Assignments Score": new_assignment_score
    }])

    predicted_marks = float(best_model.predict(new_student)[0])
    predicted_marks = float(np.clip(predicted_marks, 0, 100))

    st.metric(
        "Predicted Final Marks",
        f"{predicted_marks:.2f} / 100"
    )

    st.info(
        "This is an estimate from a model trained on synthetic data. "
        "It is not a guaranteed examination result."
    )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Student Performance Predictor | Python, Pandas, "
    "Scikit-learn, Matplotlib, Seaborn and Streamlit"
)
