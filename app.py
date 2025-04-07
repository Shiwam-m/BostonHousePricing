import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template,redirect,flash,session
from markupsafe import escape
import numpy as np 
import pandas 

app = Flask(__name__)
model = pickle.load(open('Model.pickle', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json().get('data') 
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    data_array = np.array(data).reshape(1, -1) 
    final_input = scalar.transform(data_array)
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text="The predicted house price is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)
