import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N = 500

rng = np.random.default_rng(RANDOM_STATE)

attendance = rng.uniform(55, 100, N)
study_hours = rng.uniform(1, 25, N)
previous_score = rng.uniform(35, 95, N)
assignment_avg = rng.uniform(40, 100, N)
participation = rng.uniform(20, 100, N)
sleep_hours = rng.uniform(4.5, 9, N)
previous_failures = rng.integers(0, 4, N)
internet_access = rng.integers(0, 2, N)
extracurricular = rng.integers(0, 2, N)

noise = rng.normal(0, 4.5, N)

final_score = (
    0.22 * attendance
    + 0.20 * previous_score
    + 0.18 * assignment_avg
    + 0.12 * participation
    + 1.15 * study_hours
    + 2.0 * sleep_hours
    + 2.5 * internet_access
    + 1.0 * extracurricular
    - 3.5 * previous_failures
    + noise
)

final_score = np.clip(final_score, 0, 100)

df = pd.DataFrame({
    "attendance": attendance.round(2),
    "study_hours": study_hours.round(2),
    "previous_score": previous_score.round(2),
    "assignment_avg": assignment_avg.round(2),
    "participation": participation.round(2),
    "sleep_hours": sleep_hours.round(2),
    "previous_failures": previous_failures,
    "internet_access": internet_access,
    "extracurricular": extracurricular,
    "final_score": final_score.round(2),
})

Path("data").mkdir(exist_ok=True)
df.to_csv("data/student_performance.csv", index=False)
print("Created data/student_performance.csv")
