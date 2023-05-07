import pickle

# Load the model from the saved file
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a function for making predictions
def make_prediction(soil_ph, phosphorous, nitrogen, potash, temperature, rainfall):
    # Make a prediction using the loaded model
    prediction = model.predict([[soil_ph, phosphorous, nitrogen, potash, temperature, rainfall]])
    return prediction[0]
