from flask import Flask, render_template, request, jsonify
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json
from flask_cors import CORS 
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained Prophet models for each region
with open('juba_prophet_model.json', 'r') as juba_model_file:
    juba_model = model_from_json(juba_model_file.read())

with open('yei_prophet_model.json', 'r') as yei_model_file:
    yei_model = model_from_json(yei_model_file.read())

with open('wau_prophet_model.json', 'r') as wau_model_file:
    wau_model = model_from_json(wau_model_file.read())

def forecast(model, data):
    try:
        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Rename columns to 'ds' and 'y' for Prophet
        df = df.rename(columns={'date': 'ds', 'Temperature': 'temp'})

        # Make future dataframe for prediction
        future = model.make_future_dataframe(periods=int(data['periods']), freq='M', include_history=False)
        future['Temperature'] = data['Temperature']

        # Forecast
        forecast = model.predict(future)

        # Extract relevant columns from the forecast
        forecast_data = forecast['yhat']

        # Get the last forecast result
        last_forecast_result = forecast_data.iloc[-1]

        # Convert the result to a plain integer
        result_as_integer = int(last_forecast_result)

        return result_as_integer

    except Exception as e:
        print(f'Error during forecast for model {model}: {e}')
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/juba', methods=['POST'])
def predict_juba():
    try:
        data = request.get_json()
        result = forecast(juba_model, data)
        return jsonify({'status': 'success', 'region': 'Juba', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/predict/yei', methods=['POST'])
def predict_yei():
    try:
        data = request.get_json()
        result = forecast(yei_model, data)
        return jsonify({'status': 'success', 'region': 'Yei', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/predict/wau', methods=['POST'])
def predict_wau():
    try:
        data = request.get_json()
        result = forecast(wau_model, data)
        return jsonify({'status': 'success', 'region': 'Wau', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
