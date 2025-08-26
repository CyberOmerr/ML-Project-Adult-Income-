# Task 5 – API Demo with FastAPI



from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model
model = joblib.load("adult_rf_model.joblib")

# Define API
app = FastAPI()

class InputData(BaseModel):
    age:int; workclass:str; fnlwgt:int; education:str; education_num:int
    marital_status:str; occupation:str; relationship:str; race:str; sex:str
    capital_gain:int; capital_loss:int; hours_per_week:int; native_country:str

@app.post("/predict")
def predict(data:InputData):
    df_in = pd.DataFrame([data.dict()])
    pred = model.predict(df_in)[0]
    return {"prediction": int(pred)}

@app.get("/health")
def health():
    return {"status":"ok"}

print("✅ FastAPI app defined. In Colab, run with:")
print("!uvicorn app:app --reload --host 0.0.0.0 --port 8000")
