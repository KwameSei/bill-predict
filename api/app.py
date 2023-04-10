from fastapi import FastAPI
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder  # Used to convert text to 0s and 1s
from numpy import array
from pydantic import BaseModel
import uvicorn  # Used to create a local server
import pickle


# Initializing the fast API server
app = FastAPI()
origins = [
    # "http://localhost.nathaniel.com",
    # "https://localhost.nathaniel.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a class to receive the data from the client
class Data(BaseModel):
    Appliance: list[str]
    Wats: float
    Time: float
    Days: float

# Create a route to receive the data from the client
@app.get('/')
def index():
    return {'data': 'Welcome to the electricity bill prediction API'}
 
# Create a route to receive the data from the client
@app.post('/predict-bill')
def E_bill(data: Data):
    # Load the model
    model = pickle.load(open('../ml_model/electricity-bill', 'rb'))

    encoder = OneHotEncoder()
    encoded_appliances = encoder.fit_transform([[app] for app in data.Appliance]).toarray()[0]

    # Make a prediction using the model
    prediction = model.predict([list(encoded_appliances) + [data.Wats, data.Time, data.Days]])
    return {'prediction': prediction.tolist()}

# Run the server
if __name__ == '__main__':
    uvicorn.run("app:app", host='localhost', port=8000, reload=True)
