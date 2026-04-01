from flask import Flask, render_template, request
import pickle
import pandas as pd

# Initialize app
app = Flask(__name__)

# Load trained ARIMA model
model = pickle.load(open('weather(3).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get number of days to forecast from user
        days = int(request.form['days'])

        # Forecast future values
        forecast = model.forecast(steps=days)

        # Convert result to list
        output = list(forecast)

        return render_template('index.html', prediction_text=f'Forecast: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)