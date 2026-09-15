from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "adult_income_pipeline.joblib"
)



app = FastAPI(
    title="Adult Income Prediction API",
    description=(
        "API for predicting whether annual income "
        "is above or below $50K."
    ),
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=17, le=100)
    workclass: str
    fnlwgt: int = Field(..., ge=0)
    education: str
    education_num: int = Field(..., ge=1, le=20)
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., ge=0)
    capital_loss: int = Field(..., ge=0)
    hours_per_week: int = Field(..., ge=1, le=168)
    native_country: str


class PredictionResponse(BaseModel):
    prediction: str
    probability: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}. "
            "Run src/train.py first."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True
except Exception as error:
    print(f"Failed to load model: {error}")

    model = None
    model_loaded = False


@app.get("/")
def root():

    return {
        "message": "Adult Income Prediction API",
        "docs": "/docs",
        "health": "/health",
        "prediction": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return {
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(data: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not available. "
                "Run the training pipeline first."
            ),
        )

    try:
        input_data = data.model_dump()

        input_df = pd.DataFrame(
            [input_data]
        )
        
        prediction = int(
            model.predict(input_df)[0]
        )

        probabilities = model.predict_proba(
            input_df
        )[0]

        positive_class_index = list(
            model.classes_
        ).index(1)

        probability = float(
            probabilities[positive_class_index]
        )

        label = (
            ">50K"
            if prediction == 1
            else "<=50K"
        )

        return {
            "prediction": label,
            "probability": probability,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error