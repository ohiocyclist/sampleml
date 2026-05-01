from flask import Flask, request, abort
from flask_cors import CORS
import json
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["*"]}})

@app.route('/census.json')
def censuspred():
    """
    Load a pickle of a pre-built census model predicting whether a person
    will reach >=50k of salary this year.  Built in Jupyter notebook
    XGB Adult Interpreble.
    Model parameter Workclass is optional, the others are mandatory.
    Model is histogram gradient boosting and is 84% accurate to the test
    data.  Training data is from https://archive.ics.uci.edu/dataset/2/adult

    Returns
    -------
    JSON OBJECT (STRING)
        JSON with variable 'result' describing the model's prediction

    """
    with open('histgradient.pkl', 'rb') as fh:
        histmodel = pickle.load(fh)
    with open('columntransform.pkl', 'rb') as fh:
        ct = pickle.load(fh)
    age = request.args.get('AGE', type=int, default=-1)
    if age < 0:
        return abort(400, 'Must include AGE')
    education = request.args.get('EDUCATION', type=int, default=-1)
    if education < 0:
        return abort(400, 'Must include EDUCATION')
    capgain = request.args.get('CAPGAIN', type=int, default=-1)
    if capgain < 0:
        return abort(400, 'Must include CAPGAIN')
    caploss = request.args.get('CAPLOSS', type=int, default=-1)
    if caploss < 0:
        return abort(400, 'Must include CAPLOSS')
    hours = request.args.get('HOURS', type=int, default=-1)
    if hours < 0:
        return abort(400, 'Must include HOURS')    
    workclass = request.args.get('WORKCLASS', type=str)
    nX = pd.DataFrame({'Age': [age], 'EduContinuous': [education], 
                       'CapGain': [capgain], 'CapLoss': [caploss], 
                       'Hours': [hours], 'Workclass': [' ' + str(workclass)]
                  })
    nameX = ct.transform(nX)    
    answer = histmodel.predict(nameX)
    ansdict = {'result': answer[0]}
    return json.dumps(ansdict)

if __name__ == '__main__':
    app.run(debug=False, port=8000, host='127.0.0.1')
