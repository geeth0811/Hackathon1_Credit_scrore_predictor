from fastapi.testclient import TestClient
from main import app
import datetime

# test to check the correct functioning of the /Group16 route
def test_Group16():
    with TestClient(app) as client:
        response = client.get("/Group16")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"Group16": "Members are Geeth and Ameer", "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}

# testing just whether the classes api is working fine
def test_classes():
    with TestClient(app) as client:
        response = client.get("/classes")
        # asserting the correct response status is received
        assert response.status_code == 200

# test to check if Bad risk is classified correctly
def test_pred_Bad_risk():
    # defining a sample payload for the testcase
    payload = {
        "status": 1,
        "duration": 60,
        "credit_history": 2,
        "purpose": 5,
        "amount": 100000,
        "savings": 1,
        "employment_duration": 1,
        "installment_rate": 1,
        "personal_status_sex": 1,
        "other_debtors": 1,
        "present_residence": 1,
        "property": 1,
        "age": 65,
        "other_installment_plans": 1,
        "housing": 1,
        "number_credits": 1,
        "job": 1,
        "people_liable": 1,
        "telephone": 1,
        "foreign_worker": 1,
    }
    with TestClient(app) as client:
        response = client.post("/predict_creditscore", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"Cost_Matrix_Risk": "Bad Risk", "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}


def test_pred_Good_risk():
    # defining a sample payload for the testcase
    payload = {
        "status": 4,
        "duration": 9,
        "credit_history": 4,
        "purpose": 0,
        "amount": 841,
        "savings": 1,
        "employment_duration": 2,
        "installment_rate": 4,
        "personal_status_sex": 2,
        "other_debtors": 1,
        "present_residence": 4,
        "property": 2,
        "age": 21,
        "other_installment_plans": 3,
        "housing": 1,
        "number_credits": 1,
        "job": 3,
        "people_liable": 2,
        "telephone": 1,
        "foreign_worker": 2,
    }
    with TestClient(app) as client:
        response = client.post("/predict_creditscore", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"Cost_Matrix_Risk": "Good Risk", "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}

# TC2: Here we are checking correct functioning of feedback loop by providing a valid payload.
def test_feedback_loop():
    #defining a sample payload for the testcase
    payload = [{
        "status": 2,
        "duration": 18,
        "credit_history": 4,
        "purpose": 2,
        "amount": 1049,
        "savings": 1,
        "employment_duration": 2,
        "installment_rate": 4,
        "personal_status_sex": 2,
        "other_debtors": 1,
        "present_residence": 4,
        "property": 2,
        "age": 21,
        "other_installment_plans": 3,
        "housing": 1,
        "number_credits": 1,
        "job": 3,
        "people_liable": 2,
        "telephone": 1,
        "foreign_worker": 2,
        "credit_risk": "Good Risk",
    }]
    with TestClient(app) as client:
        response = client.post("/feedback_loop", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"detail": "Feedback loop successful", "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}