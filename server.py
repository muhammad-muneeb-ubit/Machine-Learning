from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

pipeline = joblib.load(
    "models/gradient_boosting_model_v1.pkl"
)

class InputData(BaseModel):
    work_year: int
    experience_level: str
    employment_type: str
    job_title: str
    remote_ratio: int
    company_location: str
    company_size: str
    employee_residence: str

@app.post("/predict-salary")
def predict_salary(input_data: InputData):
    # Convert input data to DataFrame
    sample = pd.DataFrame([input_data.model_dump()])
    
    # Make prediction
    prediction = pipeline.predict(sample)[0]
    
    return {
        "status": "success",
        "predicted_salary": prediction.round(2)
        }