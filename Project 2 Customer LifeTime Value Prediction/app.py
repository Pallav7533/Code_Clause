from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("new_CLV.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict_clv', methods=['POST'])
def predict_clv():
    
    data = request.get_json()

    
    input_df = pd.DataFrame([data])  

    
    required_columns = ['frequency', 'recency', 'T', 'monetary_value']
    if not all(col in input_df.columns for col in required_columns):
        return jsonify({'error': 'Invalid input data'}), 400


    predictions = model.predict(input_df[required_columns])


    return jsonify({'predictions': predictions.tolist()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
