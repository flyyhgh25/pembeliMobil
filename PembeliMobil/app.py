import numpy as np
from flask import Flask, request, render_template
import pickle
from model import PrepoceesingData

flask_app = Flask(__name__)


@flask_app.route("/",methods=["POST", "GET"],endpoint='')
def home():
    dataM = PrepoceesingData()
    dataM.proses("dataset/calonpembelimobil.csv")
    dataM.DataSelection()
    result = ''
    dataInputan = []
    if request.method=='POST':
        usia = float(request.form["Usia"])
        status = float(request.form["Status"])
        kelamin = float(request.form["Kelamin"])
        penghasilan = float(request.form["Penghasilan"]) 
        memiliki_mobil = float(request.form["Memiliki_Mobil"])
        metode = request.form["metode"]
        input_features = [[usia, status, kelamin, penghasilan, memiliki_mobil]]
        if(metode=='knn'):
            dataM.MetodeKnn()
            model = pickle.load(open("modelKnnPembeli.pkl", "rb"))
            result = model.predict(input_features)[0]
            dataInputan = [usia, status, kelamin, penghasilan, memiliki_mobil,metode]
            return render_template("index.html", result=result, dataInputan=dataInputan)
        elif(metode=='random_forest'):
            dataM.MetodeRandomForestClassifier()
            model = pickle.load(open("modelRFPembeli.pkl", "rb"))
            result = model.predict(input_features)[0]
            dataInputan = [usia, status, kelamin, penghasilan, memiliki_mobil,metode]
            return render_template("index.html", result=result, dataInputan=dataInputan)
        elif(metode=='tree'):
            dataM.MetodeTree()
            model = pickle.load(open("modelTreePembeli.pkl", "rb"))
            result = model.predict(input_features)[0]
            dataInputan = [usia, status, kelamin, penghasilan, memiliki_mobil,metode]
            return render_template("index.html", result=result, dataInputan=dataInputan)
        elif(metode=='naive_bayes'):
            dataM.MetodeNaiveBayes()
            model = pickle.load(open("modelNBPembeli.pkl", "rb"))
            result = model.predict(input_features)[0]
            dataInputan = [usia, status, kelamin, penghasilan, memiliki_mobil,metode]
            return render_template("index.html", result=result, dataInputan=dataInputan)
    else:
        return render_template("index.html", result=result, dataInputan=dataInputan)

# @flask_app.route("/predict", methods=["POST", "GET"])
# def predict():
#     try:
       

#     except ValueError as e:
#         error_message = f"Error: {e}. Pastikan input valid untuk semua kolom."
#         return render_template("error.html", error_message=error_message)

#     except Exception as e:
#         error_message = f"Error: {e}. Terjadi kesalahan yang tidak terduga."
#         return render_template("error.html", error_message=error_message)

#     return render_template("index.html", result=result)

if __name__ == "__main__":
    flask_app.run(debug=True)
 