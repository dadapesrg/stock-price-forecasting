import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
import pickle
with open('results/lstm_stock_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize a MinMaxScaler (assuming the same scaler was used during training)
scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array(data['input'])
    features = np.array(input_data).reshape(-1, 1)
       
    # Transform the data to fit shape of the LSTM input)
    scaled_features = scaler.fit_transform(features)
    
    # Repeat the features until it reaches the necessary length
    features_repeated = np.tile(scaled_features, (60, 1))  # Repeats to get a shape of (60, 5)
    scaled_features = features_repeated.reshape(1, 60, 5)  # Reshpae according to model
      
    # Make predictions for 'Close' price in the 4th column
    prediction = model.predict(scaled_features)
    prediction = scaler.inverse_transform(np.zeros((1, 5)) + prediction)[0, 3]  # 'Close' price is the 4th column

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8282)

