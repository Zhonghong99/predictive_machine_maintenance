from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Model class to handle predictions
class PredictiveMaintenanceModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.is_ready = False
        self.feature_names = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Type',
            'Product ID'
        ]

    def load(self, model_path, preprocessor_path):
        try:
            print(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            print(f"Loading preprocessor from {preprocessor_path}")
            self.preprocessor = joblib.load(preprocessor_path)
            self.is_ready = True
            print("Model and preprocessor loaded successfully")
        except Exception as e:
            print(f"Error loading model or preprocessor: {str(e)}")
            raise

    def predict(self, data):
        try:
            if not self.is_ready:
                raise RuntimeError("Model not loaded")
            
            # Create test data
            test_data = pd.DataFrame({
                'Air temperature [K]': [float(data['Air temperature [K]'])],
                'Process temperature [K]': [float(data['Process temperature [K]'])],
                'Rotational speed [rpm]': [float(data['Rotational speed [rpm]'])],
                'Torque [Nm]': [float(data['Torque [Nm]'])],
                'Tool wear [min]': [float(data['Tool wear [min]'])],
                'Type': ['L'],
                'Product ID': ['M14860']
            })
            
            processed_data = self.preprocessor.transform(test_data)
            probabilities = self.model.predict_proba(processed_data)
            probability = float(probabilities[0][1])  # Convert numpy.float to Python float
            
            return {
                'prediction': bool(probability > 0.7),  # Convert numpy.bool_ to Python bool
                'probability': probability,
                'maintenance_recommended': bool(probability > 0.7)  # Convert numpy.bool_ to Python bool
            }
            
        except Exception as e:
            print(f"Error in predict method: {str(e)}")
            raise

# Initialize model container
model_service = PredictiveMaintenanceModel()

# Update the JavaScript in HTML_TEMPLATE to match exact feature names
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Predictive Maintenance System</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Roboto', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 500;
        }

        .form-group {
            margin-bottom: 25px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #34495e;
            font-weight: 500;
        }

        input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        input:focus {
            border-color: #3498db;
            outline: none;
            box-shadow: 0 0 5px rgba(52,152,219,0.3);
        }

        button {
            background: #3498db;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
        }

        button:hover {
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        #result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        .success {
            background: #d4edda;
            border-left: 5px solid #28a745;
        }

        .warning {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
        }

        .error {
            background: #f8d7da;
            border-left: 5px solid #dc3545;
        }

        h3 {
            margin-bottom: 15px;
            color: #2c3e50;
        }

        .probability-value {
            font-size: 1.2em;
            font-weight: 500;
            margin: 10px 0;
        }

        .recommendation {
            font-size: 1.1em;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Predictive Maintenance System</h1>
        <form id="predictionForm">
            <div class="form-group">
                <label>Air temperature [K]</label>
                <input type="number" name="air_temp" step="any" required>
            </div>
            <div class="form-group">
                <label>Process temperature [K]</label>
                <input type="number" name="process_temp" step="any" required>
            </div>
            <div class="form-group">
                <label>Rotational speed [rpm]</label>
                <input type="number" name="rotation_speed" step="any" required>
            </div>
            <div class="form-group">
                <label>Torque [Nm]</label>
                <input type="number" name="torque" step="any" required>
            </div>
            <div class="form-group">
                <label>Tool wear [min]</label>
                <input type="number" name="tool_wear" step="any" required>
            </div>
            <input type="hidden" name="type" value="L">
            <input type="hidden" name="product_id" value="M14860">
            <button type="submit">Predict Maintenance</button>
        </form>
        <div id="result" style="display: none;"></div>
    </div>

    <script>
        document.getElementById('predictionForm').onsubmit = function(e) {
            e.preventDefault();
            
            const formData = {
                'Air temperature [K]': parseFloat(this.air_temp.value),
                'Process temperature [K]': parseFloat(this.process_temp.value),
                'Rotational speed [rpm]': parseFloat(this.rotation_speed.value),
                'Torque [Nm]': parseFloat(this.torque.value),
                'Tool wear [min]': parseFloat(this.tool_wear.value)
            };

            console.log('Sending data to server:', formData);

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            })
            .then(response => {
                console.log('Raw response:', response);
                return response.json();
            })
            .then(data => {
                console.log('Parsed response data:', data);
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                
                if (data.error) {
                    console.error('Error from server:', data.error);
                    resultDiv.className = 'error';
                    resultDiv.innerHTML = `
                        <h3>❌ Error</h3>
                        <div class="error-message">${data.error}</div>
                    `;
                    return;
                }
                
                if (data.maintenance_recommended) {
                    resultDiv.className = 'warning';
                    resultDiv.innerHTML = `
                        <h3>⚠️ Maintenance Recommended</h3>
                        <div class="probability-value">Failure Probability: ${(data.probability * 100).toFixed(2)}%</div>
                        <div class="recommendation">Please schedule maintenance soon.</div>
                        <div class="debug-info">
                            <strong>Debug Information:</strong><br>
                            Raw probability: ${data.probability}<br>
                            Prediction: ${data.prediction}<br>
                            Input Values:<br>
                            Air temperature: ${formData['Air temperature [K]']} K<br>
                            Process temperature: ${formData['Process temperature [K]']} K<br>
                            Rotational speed: ${formData['Rotational speed [rpm]']} rpm<br>
                            Torque: ${formData['Torque [Nm]']} Nm<br>
                            Tool wear: ${formData['Tool wear [min]']} min
                        </div>
                    `;
                } else {
                    resultDiv.className = 'success';
                    resultDiv.innerHTML = `
                        <h3>✅ Equipment Healthy</h3>
                        <div class="probability-value">Failure Probability: ${(data.probability * 100).toFixed(2)}%</div>
                        <div class="recommendation">No immediate maintenance needed.</div>
                        <div class="debug-info">
                            <strong>Debug Information:</strong><br>
                            Raw probability: ${data.probability}<br>
                            Prediction: ${data.prediction}<br>
                            Input Values:<br>
                            Air temperature: ${formData['Air temperature [K]']} K<br>
                            Process temperature: ${formData['Process temperature [K]']} K<br>
                            Rotational speed: ${formData['Rotational speed [rpm]']} rpm<br>
                            Torque: ${formData['Torque [Nm]']} Nm<br>
                            Tool wear: ${formData['Tool wear [min]']} min
                        </div>
                    `;
                }
            })
            .catch(error => {
                console.error('Fetch error:', error);
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = 'error';
                resultDiv.innerHTML = `
                    <h3>❌ Error</h3>
                    <div class="error-message">Error making prediction: ${error.message}</div>
                `;
            });
        };
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = model_service.predict(data)
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_ready': model_service.is_ready})

if __name__ == '__main__':
    model_service.load('best_model.pkl', 'preprocessor.pkl')
    app.run(debug=True, port=5000)