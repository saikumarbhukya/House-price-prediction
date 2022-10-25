from flask import Flask,render_template,request
import numpy as np
import pickle
app = Flask(__name__)

@app.route('/')
def fun1():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def are():
    a = [i for i in request.form.values()]
    b = np.array(a,dtype=int)
    b = [b]
    sol = pickle.load(open('regmodel.pkl','rb'))
    prediction_value = sol.predict(b)
    prediction_value = prediction_value[0]

    return render_template('index.html',result = prediction_value)


if __name__ == '__main__':
    app.run(debug=True)