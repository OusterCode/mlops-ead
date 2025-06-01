import os
import mlflow
import logging
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from contextlib import asynccontextmanager


class Fetal_Health_Data(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model
    loaded_model = load_model()
    yield
    

app = FastAPI(
    title="Fetal Health API",
    description="API for Fetal Health Prediction",
    version="1.0.0",
    openapi_tags=[
        {"name": "Health", 
         "description": "Get API health"
         },
        {"name": "Prediction", 
         "description": "Model Prediction"}
    ],
        lifespan=lifespan
)

# Configure MLflow credentials and URI
MLFLOW_TRACKING_USERNAME = 'oustercode'
MLFLOW_TRACKING_PASSWORD = 'bd4e9dd31cf08f1fdbcac7bc95d0037c57c2850d'
MLFLOW_TRACKING_URI = 'https://dagshub.com/oustercode/puc-mlops-class.mlflow'

os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI



def load_model():

    model_name = "fetal_health"
    
    print("Setting mlflow and creating client...")
    client = mlflow.MlflowClient()
        
    print("getting registered model...")
    registered_model = client.get_registered_model(model_name)
    
    print("reading latest version...")
    model_version = registered_model.latest_versions[-1].version
    
    print("Finally loading model...")
    model_uri = f"models:/{model_name}/{model_version}"
    
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model {model_name} version {model_version} loaded sucessfully!")
    print(loaded_model)
    return loaded_model


@app.get(path ="/",
         tags=["Health"])
def api_health():
    return {"status": "healthy"}


@app.post(path="/predict",
          tags=["Prediction"])


def predict(request: Fetal_Health_Data):
      
    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations
    ]).reshape(1, -1)
    
    print(received_data)
    prediction = loaded_model.predict(received_data)
    print(f"Prediction: {prediction}")    
    return {"prediction": str(np.argmax(prediction[0]))}