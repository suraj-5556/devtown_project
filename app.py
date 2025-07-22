from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

scaler = joblib.load("scaler.pkl")
model = load_model("my_model.h5")

features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
            'sex', 'smoking', 'time']

@app.route('/')
def index():
    return render_template("index.html", features=features, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[feature]) for feature in features]
        input_data = np.array(input_data).reshape(1, -1)
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0][0]
        result = "Patient is likely to die. (1)" if prediction > 0.5 else "Patient is likely to survive. (0)"
        return render_template("index.html", features=features, prediction=result)
    except Exception as e:
        return f"<h3 style='color:red;'>Error: {e}</h3>"

if __name__ == '__main__':
    app.run(debug=True)
