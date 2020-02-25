import pickle
import json
import numpy
import pandas as pd
from sklearn.externals import joblib
from azureml.core.model import Model
import azureml.train.automl

model_name = "AutoMLbf882d31d3" # PASTE MODEL NAME HERE

def init():
    global model
    model_path = Model.get_model_path(model_name = model_name)
    model = joblib.load(model_path) # deserialize the model file back into a sklearn model

def run(rawdata):
    try:
        data = json.loads(rawdata)['data']
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result":result.tolist()})
