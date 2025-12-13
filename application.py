import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

ridge_model = pickle.load(open("models/ridgecv.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predictdata", methods=["POST"])
def predict_datapoint():
    data = [
        float(request.form["Temperature"]),
        float(request.form["RH"]),
        float(request.form["Ws"]),
        float(request.form["Rain"]),
        float(request.form["FFMC"]),
        float(request.form["DMC"]),
        float(request.form["ISI"]),
        float(request.form["Classes"]),
        float(request.form["Region"]),
    ]

    scaled = standard_scaler.transform([data])
    result = ridge_model.predict(scaled)

    return render_template("home.html", results=round(result[0], 2))

if __name__ == "__main__":
    app.run(debug=True)
