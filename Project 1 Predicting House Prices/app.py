# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


model_path = 'linear_regression_model.pkl'  
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
   
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
    bedrooms = float(request.form['bedrooms'])
    square_foot = float(request.form['square_foot'])

    features = np.array([[bedrooms, square_foot]])

    
    predicted_price = model.predict(features)[0]


    return render_template('index.html', prediction_text=f'Predicted House Price: ${predicted_price:,.2f}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  
