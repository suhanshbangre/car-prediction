#Dependencies
import car_prediction_model
from flask import Flask, jsonify, request
import joblib
import traceback
import pandas as pd
import numpy as np

#Your API definition
app = Flask(__name__)
classifier = True
@app.route('/predict', methods=['POST'])
def predict():
    global classifier
    if classifier:
        try:
            json_=request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            prediction = list(classifier.predict(query))            
            return jsonify({'prediction': str(prediction)})

        except:
            return jsonify({'trace': traceback.format_exc()})
    
    else:
        print('Train the model first')
        print('No model here to use')         
            
if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
        
    classifier = joblib.load("car_prediction_model.pkl")
    print('Model loaded')
    
    app.run(port = port, debug = True)
    