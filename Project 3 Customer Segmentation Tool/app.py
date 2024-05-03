import pickle
from flask import Flask, render_template, request, send_file, Response
import pandas as pd
import io

app = Flask(__name__)

model = r'C:\Users\U.S\Desktop\CodeClause May 2024\Project 3 Customer Segmentation Tool\goldenproject.pkl'

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/cluster', methods=['POST'])
def cluster():
    uploaded_file = request.files.get('file')
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        clusters = model.predict(data)
        data['Clusters'] = clusters

        csv_bytes = data.to_csv(index=False).encode('utf-8')
        csv_io = io.BytesIO(csv_bytes)
        csv_io.seek(0)

        
        return Response(csv_io, mimetype='text/csv', headers={"Content-Disposition": "attachment;filename=clustered_data.csv"})
    else:
        return "No file uploaded", 400

if __name__ == '__main__':
    app.run(debug=True)
