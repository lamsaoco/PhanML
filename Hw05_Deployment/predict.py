import pickle
from typing import Literal
from pydantic import BaseModel, Field

from fastapi import FastAPI
import uvicorn

class CourseLeadScoringRec(BaseModel):
    lead_source: Literal["organic_search", "social_media", "paid_ads", "referral", "events"]
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)

class PredictResponse(BaseModel):
    converted_probability: float
    converted: bool

app = FastAPI(title="course-lead-converted-prediction")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(course_lead_scoring_record):
    result = pipeline.predict_proba(course_lead_scoring_record)[0, 1]
    return float(result)

@app.post("/predict")
def predict(course_lead_scoring_record: CourseLeadScoringRec) -> PredictResponse:
    prob = predict_single(course_lead_scoring_record.model_dump())

    return PredictResponse(
        converted_probability=prob,
        converted=prob >= 0.5
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)