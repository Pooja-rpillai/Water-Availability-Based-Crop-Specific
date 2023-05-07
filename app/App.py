
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model from the saved file
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the index route
# @app.route('/')
# def index():
#     return render_template('crop.html')

import numpy as np

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pred', methods=['GET','POST'])
def pred():
    if request.method == "POST":
        soil_ph = float(request.form['soil'])
    
        phosphorous = float(request.form['php'])
        nitrogen = float(request.form['nitrogen'])
        potash = float(request.form['potash'])
        temperature = float(request.form['temperature'])
        rainfall = float(request.form['rainfall'])
        # print(nitrogen+"----------------")
        data = {
            "phosphorous" : phosphorous,
            "nitrogen" :nitrogen,
            "potash" :potash,
            "temperature" :temperature,
            "rainfall" :rainfall,
        }
        sample = np.array([[soil_ph, phosphorous, nitrogen, potash, temperature, rainfall]])
        prediction = model.predict(sample)

    else:
        pass
    return render_template("out.html",data=data,prediction= prediction)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    soil_ph = float(request.form['input2'])
    
    phosphorous = 54
    nitrogen = 26
    potash = 13
    temperature = 23
    rainfall = 53

    # Create a 2D array of the input values
    sample = np.array([[soil_ph, phosphorous, nitrogen, potash, temperature, rainfall]])

    # Make a prediction using the loaded model
    # Make a prediction using the loaded model
    #sample = [soil_ph, phosphorous, nitrogen, potash, temperature, rainfall, location]
    print(sample)
    prediction = model.predict(sample)


    # Return the prediction as a response
    print(prediction)
    print(f"The predicted crop is: {prediction[0]}")
    return render_template('result.html', prediction= prediction)
print("Open")
    # return f'The predicted crop for the given inputs is: {prediction[0]}'

print("OPen")

if __name__ == '__main__':
    app.run(debug=True)

