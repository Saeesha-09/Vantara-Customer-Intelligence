from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title = "Vantara Churn Prediction API",
    description = "API for predicting customer churn",
    version = "1.0.0"
)

saved = joblib.load("models/churn_model.joblib")
model = saved["model"]
feature_columns = saved["feature_columns"]

class CustomerData(BaseModel):
    frequency: float = Field(ge = 0)
    monetary: float
    recency: float = Field(ge = 0)
    basket_size: float = Field(ge = 0)

@app.get("/")
def home():
    return {
        "message": "Vantara Churn Prediction API is running",
        "status": "healthy"
    }

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model.named_steps["logisticregression"]).__name__,
        "features": feature_columns
    }

@app.post("/predict/churn")
def predict_churn(customer: CustomerData):
    input_data = customer.model_dump()

    input_df = pd.DataFrame(
        [[input_data[column] for column in feature_columns]],
        columns = feature_columns
    )

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    return {
        "churn_prediction": prediction,
        "churn_label": ("Likely to churn" if prediction == 1
                        else "Not predicted to churn"),
        "probability_of_churn": round(float(probabilities[1]), 4)
    }