from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = pickle.load(open('LM_MAOD_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


scaler = MinMaxScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        SeaIce=float(request.form['SeaIce'])
        Wind=float(request.form['Wind'])
        Chlorophyll = float(request.form['Chlorophyll'])
        
        env = np.array([SeaIce,Wind,Chlorophyll])
        env = env.reshape((-1,1))
        scaler = MinMaxScaler()
        scaler.fit(env)
        x_scaled = scaler.transform(env)


        prediction=model.predict(x_scaled)
        output=round(prediction[0],2)

        if output<0:
            return render_template('index.html',prediction_texts="Sorry, no prediction of marine aerosol optical depth available")
        else:
            return render_template('index.html',prediction_text="MAOD {}".format(output))

    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

