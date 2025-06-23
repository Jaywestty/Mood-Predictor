# Import Necessary Libraries

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

#Load data
df = pd.read_csv('digital_habits_vs_mental_health.csv')

#Select needed columns
num_col = ['screen_time_hours', 'social_media_platforms_used', 'hours_on_TikTok',
            'sleep_hours', 'stress_level']

#Create new multi-class target feature
def model_level(score):
    if score <= 4:
        return 'Bad'
    elif score <= 6:
        return 'Fair'
    else:
        return 'Good'
    
df['mood_level'] = df['mood_score'].apply(model_level)

#split data
X = df.drop(columns=['mood_level', 'mood_score'])
y = df['mood_level']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#build preprocessing pipeline
numerical_column = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_column, num_col)
])

#build model pipeline

model = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.1, loss_function='MultiClass',verbose=0)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('ensemble', model)
])

pipeline.fit(X_train, y_train)


#Save the model
with open('cat_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print('Model saved succesfully')

sample_data = {
    'screen_time_hours':5.3,
    'social_media_platforms_used':2,
    'hours_on_TikTok':5.3,
    'sleep_hours':7,
    'stress_level':3
    }

#create a dataframe

sample_df = pd.DataFrame([sample_data])

#Predict
pipeline.predict(sample_df)[0]

