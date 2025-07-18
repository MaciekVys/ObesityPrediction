from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("../model/model.pkl")
label_encoder = joblib.load("../model/label_encoder.pkl")


@app.route('/', methods=['GET'])
def home():
    form_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Obesity Prediction</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    * {
        box-sizing: border-box;
    }

    body {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #e0f7fa, #fce4ec);
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }

    h2 {
        margin-top: 50px;
        font-size: 32px;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
    }

    form {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
        border-radius: 20px;
        padding: 40px 30px;
        margin-top: 30px;
        max-width: 500px;
        width: 90%;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .field {
        margin-bottom: 20px;
    }

    label {
        font-weight: 600;
        margin-bottom: 6px;
        display: block;
        color: #34495e;
    }

    input[type="text"],
    select {
        width: 100%;
        padding: 12px 15px;
        border-radius: 12px;
        border: 1px solid #ccc;
        background: #fdfdfd;
        font-size: 15px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    input[type="text"]:focus,
    select:focus {
        border-color: #5dade2;
        outline: none;
        box-shadow: 0 0 4px rgba(93, 173, 226, 0.5);
        background-color: #ffffff;
    }

    input[type="submit"] {
        background: linear-gradient(135deg, #5dade2, #3498db);
        color: white;
        font-weight: bold;
        border: none;
        padding: 14px;
        font-size: 16px;
        width: 100%;
        border-radius: 14px;
        cursor: pointer;
        transition: background 0.3s ease, transform 0.2s ease;
        margin-top: 10px;
    }

    input[type="submit"]:hover {
        background: linear-gradient(135deg, #2e86de, #2980b9);
        transform: scale(1.02);
    }

    @media (max-width: 600px) {
        h2 {
            font-size: 24px;
            margin-top: 30px;
        }

        form {
            padding: 30px 20px;
        }
    }
</style>

    </head>
    <body>
        <h2>Predict Obesity</h2>
        <form action="/prediction" method="post">
            <div class="field">
                <label>Gender:</label><br>
                <select name="Gender">
                    <option value="0">Male</option>
                    <option value="1">Female</option>
                </select>
            </div>

            <div class="field">
                <label>Age:</label><br>
                <input type="text" name="Age" required>
            </div>

            <div class="field">
                <label>Height (m):</label><br>
                <input type="text" name="Height"required>
            </div>

            <div class="field">
                <label>Weight (kg):</label><br>
                <input type="text" name="Weight"required>
            </div>

            <div class="field">
                <label>Family History of Overweight:</label><br>
                <select name="family_history">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="field">
                <label>FAVC (High Caloric Food):</label><br>
                <select name="FAVC">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="field">
                <label>FCVC (Vegetable Consumption):</label><br>
                <select name="FCVC">
                    <option value="4">Much</option>
                    <option value="3">Noraml</option>
                    <option value="2">Little</option>
                    <option value="1">Few</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="field">
                <label>NCP (Number of Main Meals):</label><br>
                <input type="text" name="NCP"required>
            </div>

            <div class="field">
                <label>CAEC (Food Between Meals):</label><br>
                <select name="CAEC">
                    <option value="0">No</option>
                    <option value="1">Sometimes</option>
                    <option value="2">Frequently</option>
                    <option value="3">Always</option>
                </select>
            </div>

            <div class="field">
                <label>SMOKE:</label><br>
                <select name="SMOKE">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="field">
                <label>CH2O (Water Consumption in Liters):</label><br>
                <input type="text" name="CH2O" required>
            </div>

            <div class="field">
                <label>SCC (Calories Monitor):</label><br>
                <select name="SCC">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="field">
                <label>FAF (Physical Activity Frequency):</label><br>
                <select name="FAF">
                    <option value="3">High activity</option>
                    <option value="2">Average activity</option>
                    <option value="1">Low activity</option>
                    <option value="0">Lack of activity</option>
                </select>
            </div>

            <div class="field">
                <label>TUE (Time Using Technology per Day):</label><br>
                <input type="text" name="TUE"required>
            </div>

            <div class="field">
                <label>CALC (Alcohol Consumption):</label><br>
                <select name="CALC">
                    <option value="0">No</option>
                    <option value="1">Sometimes</option>
                    <option value="2">Frequently</option>
                    <option value="3">Always</option>
                </select>
            </div>

            <div class="field">
                <label>MTRANS (Transport Mode):</label><br>
                <select name="MTRANS">
                    <option value="0">Automobile</option>
                    <option value="1">Bike</option>
                    <option value="2">Motorbike</option>
                    <option value="3">Public_Transportation</option>
                    <option value="4">Walking</option>
                </select>
            </div>

            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    """
    return render_template_string(form_html)


@app.route('/prediction', methods=['POST'])
def predict():
    try:
        features_names = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC',
            'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
            'CALC', 'MTRANS'
        ]

        features_dict = {}

        for name in features_names:
            value = request.form.get(name)
            if value is None:
                return f"<h3>Error: Missing value for {name}</h3>"
            try:
                val_float = float(value)
                val_final = int(val_float) if val_float.is_integer() else val_float
                features_dict[name] = [val_final]
            except:
                return f"<h3>Error: Invalid value for {name}</h3>"

        features_df = pd.DataFrame(features_dict)
        pred_encoded = model.predict(features_df)
        prediction_label = label_encoder.inverse_transform(pred_encoded)[0]

        result_page = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    font-family: 'Inter', sans-serif;
                    background: linear-gradient(135deg, #e0c3fc, #8ec5fc);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0;
                }}
                .result-box {{
                    background: rgba(255, 255, 255, 0.3);
                    backdrop-filter: blur(12px);
                    padding: 40px 50px;
                    border-radius: 20px;
                    text-align: center;
                    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
                    border: 1px solid rgba(255,255,255,0.3);
                }}
                h2 {{
                    font-size: 28px;
                    color: #2c3e50;
                    margin-bottom: 20px;
                }}
                .label {{
                    font-size: 24px;
                    font-weight: 600;
                    background: #3498db;
                    color: white;
                    padding: 12px 20px;
                    border-radius: 10px;
                    display: inline-block;
                }}
                .back-btn {{
                    margin-top: 30px;
                    display: inline-block;
                    text-decoration: none;
                    color: #3498db;
                    font-weight: bold;
                    border: 2px solid #3498db;
                    padding: 10px 20px;
                    border-radius: 8px;
                    transition: all 0.3s ease;
                }}
                .back-btn:hover {{
                    background-color: #3498db;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="result-box">
                <h2>Your predicted obesity category is:</h2>
                <div class="label">{prediction_label}</div>
                <br>
                <a class="back-btn" href="/">← Try Again</a>
            </div>
        </body>
        </html>
        """
        return render_template_string(result_page)

    except Exception as e:
        return f"<h3>Unexpected Error: {str(e)}</h3>"



    
if __name__ == '__main__':
    app.run(debug=True)
    