import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

with open("your_model_filename.pkl", "rb") as file:
    model = pickle.load(file)


class InputData(BaseModel):
    rainfall: float


@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}


@app.post("/predict")
def predict(data: InputData):
    try:
        # Extract rainfall value from the Pydantic model
        rainfall = data.rainfall
        print(rainfall)

        # Make a prediction using the loaded model
        prediction = model.predict(
            np.array([[rainfall]])
        )  # Reshape to match the model's input shape

        print(prediction)
        # Use an array to store the +- 20% range of the prediction
        # prediction_range = np.array([[prediction[0] * 0.8, prediction[0] * 1.2]])

        # print(prediction_range)
        # Return the prediction as JSON

        # Truncate the values by 2 decimal places
        predictionMin = round(prediction[0] * 0.8, 2)
        predictionMax = round(prediction[0] * 1.2, 2)
        print(predictionMin)
        print(predictionMax)
        result = {
            "predictionMin": np.round(predictionMin, 2),
            "predictionMax": np.round(predictionMax, 2),
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
