# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from flask import Flask, request,render_template
import pickle

#Create the Application
app = Flask(__name__)

#Loading the model
model = pickle.load(open('model.pkl', 'rb'))


#Display the html interface(codes in another page)
@app.route('/')
def home():
    return render_template('index.html')

dropdown_mappings = {
    "Credit_History": {
        "1.0" : 1.0,
        "0.0" : 0.0
            
    },
    
    "Property_Area": {
        "1.0" : 1.0,
        "0.5" : 0.5,
        "0.0" : 0.0
    },
    
    "Married": {
        "1.0" : 1.0,
        "0.0" : 0.0
            
    },
    
    "Gender": {
        "1.0" : 1.0,
        "0.0" : 0.0
            
    },
    
    "Dependents":{
        "0.0" : 0.0,
        "1.0" : 1.0,
        "2.0" : 2.0,
        "3.0" : 3.0    
    },
    
    "Education":{
        "0.0" : 0.0,
        "1.0" : 1.0
           
    }
    
    
}

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    form_values = request.form.to_dict()
    
    for dropdown_name, dropdown_mapping in dropdown_mappings.items():
        selected_value = form_values.get(dropdown_name, "")
        form_values[dropdown_name] = dropdown_mapping.get(selected_value, 0.0)
    
    
    
    for key, value in form_values.items():
       form_values[key] = float(value)
    #int_features = [int(x) for x in request.form.values()]
    
    
    #final_features = [np.array(int_features)]    
    final_features = np.array(list(form_values.values())).reshape(1, -1)
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Loan Status is: {}'.format(output))

if __name__ == "__main__":
    app.run(port = 5000, debug=True)
