from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import pickle
from prophet import Prophet
from flask_cors import CORS
import logging
from prophet.serialize import model_from_json
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# Load pre-trained models
MODELS = {
    'juba': {
        'Prophet': model_from_json(open('juba_prophet_model.json', 'r').read()),
        'ARIMA': pickle.load(open('juba_arima_model.pkl', 'rb')),
        'NeuralProphet': pickle.load(open('juba_np_model.pkl', 'rb')),
        'ExponentialSmoothing': pickle.load(open('juba_es_model.pkl', 'rb'))
    },
    'yei': {
        'Prophet': model_from_json(open('yei_prophet_model.json', 'r').read()),
        'ARIMA': pickle.load(open('yei_arima_model.pkl', 'rb')),
        'NeuralProphet': pickle.load(open('yei_np_model.pkl', 'rb')),
        'ExponentialSmoothing': pickle.load(open('yei_es_model.pkl', 'rb'))
    },
    'wau': {
        'Prophet': model_from_json(open('wau_prophet_model.json', 'r').read()),
        'ARIMA': pickle.load(open('wau_arima_model.pkl', 'rb')),
        'NeuralProphet': pickle.load(open('wau_np_model.pkl', 'rb')),
        'ExponentialSmoothing': pickle.load(open('wau_es_model.pkl', 'rb'))
    }
}

def forecast(region, model_type, data):
    """
    Perform forecasting using the specified model and region.
    """
    try:
        # Validate input data
        required_keys = ['date', 'Temperature', 'Rainfall', 'periods']
        if not all(key in data for key in required_keys):
            raise ValueError("Input data must contain 'date', 'Temperature', 'Rainfall', and 'periods'.")

        # Check if model exists
        model = MODELS[region].get(model_type)
        if not model:
            raise ValueError(f"Model '{model_type}' not available for region '{region}'.")

        # Prepare data for prediction
        if model_type == 'Prophet':
            df = pd.DataFrame([data]).rename(columns={'date': 'ds', 'Temperature': 'temp', 'Rainfall': 'rain'})
            future = model.make_future_dataframe(periods=int(data['periods']), freq='M', include_history=False)
            future['temp'] = data['Temperature']
            future['rain'] = data['Rainfall']
            forecast_result = model.predict(future)
            forecast_df = forecast_result[['ds', 'yhat']]  # Extract relevant forecast results
        elif model_type == 'ARIMA':
            forecast_series = model.forecast(steps=int(data['periods']))
            forecast_df = pd.DataFrame({
                'ds': pd.date_range(start=data['date'], periods=int(data['periods']), freq='M'),
                'yhat': forecast_series
            })
        elif model_type == 'NeuralProphet':
            forecast_df = model.predict(pd.DataFrame({'ds': pd.date_range(start=data['date'], periods=int(data['periods']), freq='M')}))
        elif model_type == 'ExponentialSmoothing':
            forecast_series = model.forecast(steps=int(data['periods']))
            forecast_df = pd.DataFrame({
                'ds': pd.date_range(start=data['date'], periods=int(data['periods']), freq='M'),
                'yhat': forecast_series
            })

        # Generate Plot
        plt.figure(figsize=(10, 5))
        plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast')
        plt.xlabel('Date')
        plt.ylabel('Malaria Cases')
        plt.title(f"{region.capitalize()} ({model_type}) Forecast")
        plt.legend()
        plt.grid(True)

        # Convert plot to base64 for rendering in HTML
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return forecast_df, plot_url
    except Exception as e:
        logging.error(f"Error during forecasting: {e}")
        raise ValueError(f"Error during forecasting: {str(e)}")

@app.route('/')
def index():
    """
    Render the main HTML page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Generalized route for predicting malaria cases for a selected region and model.
    """
    try:
        data = request.get_json()
        region = data.get('region', '').lower()
        model_type = data.get('model_type', '')

        if region not in MODELS:
            return jsonify({'status': 'error', 'message': f"Invalid region '{region}'."}), 400

        forecast_df, plot_url = forecast(region, model_type, data)

        # Return JSON response with the plot and option to download data
        return jsonify({
            'status': 'success',
            'region': region.capitalize(),
            'model': model_type,
            'forecast': forecast_df.to_dict(orient='records'),
            'plot_url': f"data:image/png;base64,{plot_url}"
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download', methods=['POST'])
def download():
    """
    Endpoint to download forecast results as a CSV file.
    """
    try:
        data = request.get_json()
        forecast_df = pd.DataFrame(data['forecast'])
        csv = forecast_df.to_csv(index=False)
        response = make_response(csv)
        response.headers['Content-Disposition'] = 'attachment; filename=forecast.csv'
        response.headers['Content-Type'] = 'text/csv'
        return response
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

