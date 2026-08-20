# AI-Driven Student Performance Prediction System

An end-to-end machine learning project that predicts a student's final academic performance from attendance, study habits, previous scores, assignment performance, and other learning indicators.

## Features
- Data preprocessing and feature engineering
- Random Forest regression model
- Automatic model training and evaluation
- Interactive Streamlit web application
- Synthetic demo dataset generator
- Feature importance visualization
- Prediction with confidence-style performance bands
- Easy GitHub-ready project structure

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit, Joblib

## Project Structure
```text
ai-student-performance-prediction/
├── app.py
├── train_model.py
├── generate_data.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── student_performance.csv
├── models/
│   └── student_performance_model.joblib
└── notebooks/
    └── README.md
```

## Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-student-performance-prediction.git
cd ai-student-performance-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate demo data
```bash
python generate_data.py
```

### 4. Train the model
```bash
python train_model.py
```

### 5. Launch the app
```bash
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Input Features
- Attendance percentage
- Study hours per week
- Previous exam score
- Assignment average
- Participation rate
- Sleep hours
- Previous failures
- Internet access
- Extracurricular activity

## Output
The system predicts the expected final score and assigns a performance band:
- Excellent: 85–100
- Good: 70–84
- Average: 50–69
- At Risk: below 50

## Important Note
The included dataset is synthetic and intended for demonstration/academic use. Predictions should not be used as the sole basis for high-impact decisions about real students.

## Future Enhancements
- XGBoost/LightGBM comparison
- Explainable AI with SHAP
- Early-warning notifications
- Teacher/admin dashboard
- Student-specific recommendations
- Database integration
- Authentication and role-based access
- Deployment on Streamlit Community Cloud
