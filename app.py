import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin',
                     'BMI','DiabetesPedigreeFunction','Age']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "Patient has Diabetes"
    else:
        res_val = "Patient does not have Diabetes"
        

    return render_template('submit.html', prediction_text=res_val,glucose=input_features[1],age=input_features[7],bloodpressure=input_features[2],skinthickness=input_features[3],insulin=input_features[4],bmi=input_features[5],diabetespedigreefunction=input_features[6],pregnencies=input_features[0])

if __name__ == "__main__":
    app.run(debug=True)
