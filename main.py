import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_model, predict, retrain
from typing import List
import datetime

# defining the main app
app = FastAPI(title="Credit Score Predictor Version by Group16 (Ameer and Geeth).", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", load_model)

# class which is expected in the payload
class QueryIn(BaseModel):
    status: int
    duration: int
    credit_history: int
    purpose: int
    amount:int
    savings:int
    employment_duration:int
    installment_rate:int
    personal_status_sex:int
    other_debtors:int
    present_residence:int
    property:int
    age:int
    other_installment_plans:int
    housing:int
    number_credits:int
    job:int
    people_liable:int
    telephone:int
    foreign_worker:int


# class which is returned in the response
class QueryOut(BaseModel):
    credit_risk: str
    #adding the timestamp as an attribute in the output model
    run_Time_timestamp : str

# class which is expected in the payload while re-training
class FeedbackIn(BaseModel):
    status: int
    duration: int
    credit_history: int
    purpose: int
    amount:int
    savings:int
    employment_duration:int
    installment_rate:int
    personal_status_sex:int
    other_debtors:int
    present_residence:int
    property:int
    age:int
    other_installment_plans:int
    housing:int
    number_credits:int
    job:int
    people_liable:int
    telephone:int
    foreign_worker:int
    credit_risk: int


# Route definitions
@app.get("/Group16")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"Group16": "Members are Geeth and Ameer", "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}

@app.get("/classes")
# Healthcheck route to ensure that the API is up and running
def classes():
    return {
        'status':{1:"no checking account",2:"<0 DM", 3: "0<= ... < 200 DM",4:"... >= 200 DM / salary for at least 1 year"},
        'duration': "duration of credit month",
        'credit_history':{0:"delay in paying off in the past",1 : "critical account/other credits elsewhere",2:"no credits taken/all credits paid back duly",3:"existing credits paid back duly till now",4:"all credits at this bank paid back duly"},
        'purpose':{0 : "others", 1 : "car (new)", 2 : "car (used)", 3 :"furniture/equipment" , 4 : "radio/television", 5 : "domestic appliances", 6 : "repairs", 7 : 'education',8 : 'vacation',9 : 'retraining',10 : 'business'},
        'amount' : "credit amount in euros",
        'savings':{1 : "unknown/no savings account",2 :"... <  100 DM",3 : "100 <= ... <  500 DM",4 :"500 <= ... < 1000 DM", 5 :"... >= 1000 DM"},
        'employment_duration':{5:">=7 years", 4:"4<= <7 years",  3:"1<= < 4 years", 2:"<1 years",1:"unemployed"},
        'installment_rate':{1 :" >= 35 ",2 : "25 <= ... < 35",3 :" 20 <= ... < 25",4 :" < 20" },
        'personal_status_sex':{ 1:"male : divorced/separated",2:"female : non-single or male : single",3:"male : married/widowed", 4:"female : single"},
        'other_debtors':{1:"none", 2:"co-applicant", 3:"guarantor"},
        'present_residence' : "Duration in present residence",
        'property':{4:"real estate", 3:"building soc. savings agr./life insurance", 2:"car or other", 1:"unknown / no property"},
        'age' : "age in years",
        'other_installment_plans':{3:"none", 2:"stores", 1:"bank"},
        'housing':{1:"for free", 3:"own", 2:"rent"},
        'number_credits' : "number of credits at this bank",
        'job':{4:"management/ highly qualified employee", 3:"skilled employee / official", 2:"unskilled - resident", 1:"unemployed/ unskilled  - non-resident"},
        'people_liable' : "number of dependents",
        'telephone':{2:"yes", 1:"none"},
        'foreign_worker':{1:"yes", 2:"no"}
            }


@app.post("/predict_creditscore", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the flower_class predicted (200)
def predict_creditscore(query_data: QueryIn):
    return {"credit_risk": predict(query_data), "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}

@app.post("/feedback_loop", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct Cost_Matrix_Risk class
# Response: Dict with detail confirming success (200)
def feedback_loop(data: List[FeedbackIn]):
    retrain(data)
    return {"detail": "Feedback loop successful", "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}

# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)