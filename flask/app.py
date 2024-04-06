import numpy as np
import pickle
import pandas as pd
from flask import Flask, render_template, request

model = pickle.load(open(r"fraud.pkl", 'rb'))
app = Flask(__name__)

@app.route("/")
def about():
    return render_template('home.html')

@app.route("/home")
def about1():
    return render_template('home.html')

@app.route("/predict")
def home1():
    return render_template('predict.html')

@app.route("/pred", methods=['POST', 'GET'])
def predict():
    x = [[x for x in request.form.values()]]
    print(x)

    x = np.array(x)
    print(x.shape)

    print(x)
    pred = model.predict(x)
    print(pred[0])
    return render_template('submit.html', prediction_str=str(pred))

@app.route("/submit", methods=['POST'])
def submit():
    # Retrieve form data
    step = request.form.get('step')
    Type = request.form.get('Type')
    amount = request.form.get('amount')
    oldbalanceorg = request.form.get('oldbalanceorg')
    newbalanceorig = request.form.get('newbalanceorig')
    oldbalancedest = request.form.get('oldbalancedest')
    newbalancedest = request.form.get('newbalancedest')

    # Process the form data (add your processing code here)
    # For example, you can pass the form data to your model for prediction
    form_data = [[step, Type, amount, oldbalanceorg, newbalanceorig, oldbalancedest, newbalancedest]]
    form_data = np.array(form_data).astype(float)
    prediction = model.predict(form_data)
    prediction_str = str(prediction[0])

    return render_template('submit.html', prediction_str=prediction_str)

if __name__ == "__main__":
    app.run(debug=True)
