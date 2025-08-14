# Mood Predictor Based on Screen Time

This project predicts a user's mood score (`Good`, `Fair`, or `Bad`) based on their digital habits and screen time patterns.  
It uses machine learning models, including **CatBoost**, **Random Forest**, and **Logistic Regression**, to analyze the relationship between screen usage and mental health.

## Key Points
- Dataset: `digital_habits_vs_mental_health.csv`
- Best model: **CatBoost Classifier** (balanced performance across all mood classes)
- Preprocessing includes scaling, encoding, and SMOTE for class balancing.
- Evaluation metrics: Accuracy, Confusion Matrix, Classification Report.

## Live App
You can try the live version of the app here:  
[**Click to open the app**](https://mood-predictor-99wn.onrender.com)

## Installation
```bash
pip install -r requirements.txt
