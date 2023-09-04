import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
best_model = joblib.load('best_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict_churn():
    try:
        # Get input data from POST request
        input_data = request.get_json()
        
        input_df = pd.DataFrame(input_data)
        input_df_encoded = pd.get_dummies(input_df, columns=['Gender', 'Location'])
        input_scaled = scaler.transform(input_df_encoded)
        
        predictions = best_model.predict(input_scaled)
        
        # Return predictions as JSON response
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
