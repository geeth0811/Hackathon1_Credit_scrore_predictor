# Hackathon assignment by Geethanadh Sunkara And Ameer 

This repository contains code which demonstrates ML-Ops using a `FastAPI` application which predicts the Credit Score using the german dataset which is present in the SouthGermanCredit folder.

## Running Instructions
- Create a fork of the repo using the `fork` button if you want to contribute.
- Clone your fork using `git clone https://github.com/geeth0811/Credit_score_predictor`
- Install dependencies using `pip install -r requirements.txt`
- Run application using `python main.py`
- Run tests using `pytest`

## CI/CD
- `build` (test) for all the pull requests
- `build` (test) and `upload_zip` for all pushes

## File contents
- main.py 
- ml_utils.py
- requirements.txt
- cicd.yml
- test_app.py
- SouthGermanCredit folder where the dataset is present.
- Hackathon_CreditScoring.ipynb for checking on the dataset and decide the ML models.
