from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# step 1: load your model
model = joblib.load('boston_model.pkl')

# step 2: initialize the app
app = FastAPI(title="Boston House Price Prediction")

# step 3: create the input schema
class HouseFeatures(BaseModel):
    CRIM: float      # per capita crime rate by town
    CHAS: int       # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    NOX: float             # nitric oxides concentration (parts per 10 million)
    RM: float       # average number of rooms per dwelling
    DIS: float            # weighted distances to five Boston employment centres
    RAD: int             # index of accessibility to radial highways        
    PTRATIO: float         # pupil-teacher ratio by town
    LSTAT: float          # % lower status of the population

# step 4: create the root endpoint
@app.get("/")         # when a user visits the main homepage, run this funtion
def read_root():
    return {"message": "Welcome to the Boston House Price Prediction API!"}

# step 5: create the prediction endpoint
@app.post("/predict")      # when a user sends a POST request to the /predict endpoint, run this function
def predict_house_price(features: HouseFeatures):
    # step 5.1: convert the input data to a numpy array 
    input_data = np.array([[features.CRIM, features.CHAS, features.NOX, features.RM, features.DIS, features.RAD, features.PTRATIO, features.LSTAT]])

    # step 5.2: make the prediction using the loaded model
    log_price_prediction = model.predict(input_data)[0]   # the 0 returns the exact value rather than a list

    # step 5.3: convert the log price prediction back to the original scale
    actual_prediction = np.exp(log_price_prediction)   # convert log price back to actual price

    # step 5.4: convert to $1000
    final_price = actual_prediction * 1000

    return {"Estimated House Price": f"${final_price:,.2f}"}

    
    
    


