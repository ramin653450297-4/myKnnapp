from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('decision_tree_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        weight = float(request.form['weight'])
        height = int(request.form['height'])
        systolic_bp = int(request.form['systolic_bp'])
        diastolic_bp = int(request.form['diastolic_bp'])
        heart_rate = int(request.form['heart_rate'])
        
        # Prepare data for prediction
        input_data = pd.DataFrame([[age, weight, height, systolic_bp, diastolic_bp]], 
                                  columns=['Age', 'Weight', 'Height', 'Systolic_BP', 'Diastolic_BP','Heart_Rate'])
        
        # Predict hypertension
        prediction = model.predict(input_data)[0]
        result = 'Yes' if prediction == 1 else 'No'

        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
