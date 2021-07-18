import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

h2o.init()

#Using the South German Credit (UPDATE) Data Set
#Dataset Link : https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29

dataset = "./SouthGermanCredit/SouthGermanCredit.asc"
df = h2o.import_file(dataset)

# Reponse column
y = "credit_risk"

# Split into train & test
splits = df.split_frame(ratios = [0.8], seed = 1)
train = splits[0]
test = splits[1]

# Run AutoML for 1 minute
aml = None

# # define a RandomForestClassifier classifier
# clf = RandomForestClassifier(n_estimators = 60, max_depth = 12, random_state = 7)

# define the class encodings and reverse encodings
classes = {0: "Bad Risk", 1: "Good Risk"}
risk_threshold = 0.50

# function to train and load the model during startup
def load_model():
    global aml
    aml = H2OAutoML(max_runtime_secs=60, seed=1)
    train_model()
    
def train_model():
    aml.train(y=y, training_frame=train)
    print("Auto ML Model Training Complete")

# function to predict the Credit score using the model
def predict(query_data):
    x = list(query_data.dict().values())
    columns = list(query_data.dict().keys())
    x = h2o.H2OFrame(x).transpose()
    x.columns = columns
    prediction = aml.leader.predict(x)[0,0]
    if(prediction > risk_threshold):
        print(f"Model prediction: {classes[1]}")
        return classes[1]
    else:
        print(f"Model prediction: {classes[0]}")
        return classes[0]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    global train
    for ds in data:
        x = list(ds.dict().values())
        columns = list(ds.dict().keys())
        x = h2o.H2OFrame(x).transpose()
        x.columns = columns
        train += x
    print("Added data to the training set")
    print("Retraining the Model")
    train_model()
