from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

cols = ['category',
        'gender',
        'enable_mobile_banking',
        'enable_internet_banking',
        'enable_international_transaction',
        'account_opening_source',
        'account_open_year',
        'region']

model = pickle.load(open('customer.sav', 'rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():

    gender = request.form.get('gender', False)
    category = request.form.get('category', False)
    mobile_banking = request.form.get('mobile_banking', False)
    internet_banking = request.form.get('internet_banking', False)
    international = request.form.get('international', False)
    source = request.form.get('source', False)
    year = request.form.get('year', False)
    region = request.form.get('region', False)

    features = [gender, category, mobile_banking, internet_banking, international, source, year, region]

    print(features)
    final = np.array(features)
    input = pd.DataFrame([final], columns=cols)
    prediction = model.predict(input)[0]

    return render_template('home.html',pred='The customer is mostly likely to purchase {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = model.predict(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
