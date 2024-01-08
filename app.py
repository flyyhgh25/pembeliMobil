import numpy as np
from flask import Flask, request, render_template
import pickle
from model import DataMining

flask_app = Flask(__name__)

@flask_app.route("/", methods=["POST", "GET"], endpoint='')
def home():
    dataMining = DataMining()
    dataMining.PrepocessingData("dataset/calonpembelimobil.csv")    
    dataMining.MetodeRandomForest()
    dataMining.MetodeNaiveBayes()
    dataMining.MetodeXGBoost()
    result = []
    dataInputan = []
    if request.method == 'POST':
        try:
            usia = int(request.form["Usia"])
            kelamin = int(request.form["Kelamin"])
            penghasilan = int(request.form["Penghasilan"])
            memiliki_mobil = int(request.form["Memiliki_Mobil"])
            dataInputan = [usia, kelamin, penghasilan, memiliki_mobil]
            input_features = [[usia, kelamin, penghasilan, memiliki_mobil ]]
            # XGBoost
            modelXgb = pickle.load(open("model/modelXgBostPembeli.pkl", "rb"))
            xgb = modelXgb.predict(input_features)[0]
            # Random Forest
            modelRf = pickle.load(open("model/modelRFPembeli.pkl", "rb"))
            rf = modelRf.predict(input_features)[0]
            # Naive Bayes
            modelGnb = pickle.load(open("model/modelNBPembeli.pkl", "rb"))
            gnb = modelGnb.predict(input_features)[0]
            result = [gnb, rf, xgb]
            print('hasil keseluruhan', result)
        except Exception as e:
            print(f"Error: {str(e)}")
        return render_template("index.html", result=result, dataInputan=dataInputan)
    else:
        return render_template("index.html", result=result, dataInputan=dataInputan)

if __name__ == "__main__":
    flask_app.run(debug=True)
