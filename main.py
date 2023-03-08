from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

@app.route("/", methods = ["GET"])
def home():
    return """
    <h1>API para generar MPG prediccione</h1>
    La ruta para hacer las predicciones es /api/v0/predicciones
    argumentos de entrada:
    <ul>
    <li>
    'cylinders' - num cilindros (int)
    </li>
        <li>
    'displacement' - carrera (float)
    </li>
        <li>
    'horsepower' - caballos potencia (float)
    </li>
        <li>
    'weight' - peso en libras (float)
    </li>
        <li>
    'acceleration' - aceleracion en millas/hora**2 (float)
    </li>
        <li>
    'model_year' - año fabricacion (int)
    </li>
        <li>
    'origin' - una de [usa, japan, europe] (string)
    </li>
    </ul>
    """

@app.route("/api/v0/predicciones", methods = ["GET"])
def predictions():

    filename = 'mpg.model'
    loaded_model = pickle.load(open(filename, 'rb'))

    map_origin = {"usa":1, "japan":2, "europe": 3}

    cylinders = request.args["cylinders"]
    displacement = request.args["displacement"]
    horsepower = request.args["horsepower"]
    weight = request.args["weight"]
    acceleration = request.args["acceleration"]
    model_year = request.args["model_year"]
    origin = request.args["origin"]

    origin = map_origin[origin]

    data = [[ cylinders, displacement, horsepower, weight, acceleration,
       model_year, origin]]
    
    return jsonify(loaded_model.predict(data)[0])



if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
