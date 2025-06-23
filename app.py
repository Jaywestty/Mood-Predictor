from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle  
import os

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('cat_model.pkl', 'rb'))  

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Define features as a dictionary (with keys matching training column names)
        input_dict = {
            'screen_time_hours': float(request.form['screen_time_hours']),
            'social_media_platforms_used': int(request.form['social_media_platforms_used']),
            'hours_on_TikTok': float(request.form['hours_on_TikTok']),
            'sleep_hours': float(request.form['sleep_hours']),
            'stress_level': int(request.form['stress_level'])
        }

        input_df = pd.DataFrame([input_dict])

        prediction = str(model.predict(input_df)[0]).strip("[]'\"") # adjust if it's multi-class

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
