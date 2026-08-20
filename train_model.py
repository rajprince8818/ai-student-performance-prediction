from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/student_performance.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "student_performance_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURES = [
    "attendance",
    "study_hours",
    "previous_score",
    "assignment_avg",
    "participation",
    "sleep_hours",
    "previous_failures",
    "internet_access",
    "extracurricular",
]

TARGET = "final_score"

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

metrics = {
    "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
    "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 3),
    "r2": round(float(r2_score(y_test, predictions)), 3),
}

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(
    {"model": model, "features": FEATURES},
    MODEL_PATH,
)

METRICS_PATH.write_text(json.dumps(metrics, indent=2))

print("Model saved to:", MODEL_PATH)
print("Metrics:", metrics)
