# -*- coding: utf-8 -*-

from flask import Flask, request
import pickle
import pandas as pd

app = Flask(__name__)

@app.route("/predict/")
def predict():
    atest_encoded = pd.read_csv('atest_encoded.csv')
    model = pickle.load(open('rf_for_deployment','rb'))
    idClient = int(request.args.get('idClient'))
    
    prev = model.predict(pd.DataFrame(atest_encoded, index=[idClient]))
    proba = model.predict_proba(pd.DataFrame(atest_encoded, index=[idClient]))[0][prev[0]].round(4)
    
    if proba > 0.55:   # Seuil optimal
        lib = "Défaut"
    else:
        lib = "Sans Défaut"
    return lib


if __name__ == "__main__":
    app.run(debug=True)
