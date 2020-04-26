#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, flash, redirect, url_for, render_template, request, session, abort

#creating instance of the class
app=Flask(__name__)

app.config['SECRET_KEY'] = os.urandom(12)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

#prediction function
def ValuePredictor(to_predict_list):
    try:  
        to_predict = np.array(to_predict_list).reshape(1,12)
        loaded_model = pickle.load(open("model_yatin.pkl","rb"))
        result = loaded_model.predict(to_predict)
        return result[0]
    except Exception as e:
        print("Exception occured",e)


@app.route('/',methods = ['POST'])
def result():
    try:
        if request.method == 'POST':
            to_predict_list = request.form.to_dict()
            print(to_predict_list)

            to_predict_list=list(to_predict_list.values())
            print("==============\n",to_predict_list)

            to_predict_list = list(map(int, to_predict_list))
            print("==============\n",to_predict_list)

            result = ValuePredictor(to_predict_list)
            print("==============Result \n",result)

            if int(result)==1:
                prediction='Income more than 50K'
            else:
                prediction='Income less that 50K'

            return render_template("index.html",prediction=prediction)
    except Exception as e:
        print("Exception occured result",e)  

if __name__ == "__main__":
    app.run(debug=True)
