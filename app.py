import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## load the model
regressionmodel=pickle.load(open('regressionmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    data = request.json['data']

    features = [
        data['crim'],
        data['zn'],
        data['indus'],
        data['chas'],
        data['nox'],
        data['rm'],
        data['age'],
        data['dis'],
        data['rad'],
        data['tax'],
        data['ptratio'],
        data['b'],
        data['lstat']
    ]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=regressionmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regressionmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="the house price prediciton is{}".format(output))

if __name__=="__main__":
    app.run(debug=True)
