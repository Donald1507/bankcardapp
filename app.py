import csv
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
features_list = ['Gender','Marital_Status','Income_Category','Card_Category','Months_Inactive_12_mon','Avg_Utilization_Ratio','Total_Relationship_Count']

def encodage(dataframe):
    code = {
        'Attrited Customer':1, 'Existing Customer':0,
        'F':0, 'M':1,
        'Single':0, 'Married':1, 'Divorced':2, 
        'Less than $40K':0, '$40K - $60K':1,'$60K - $80K':2, '$80K - $120K':3, '$120K +':4,
        'Blue':0, 'Silver':1, 'Gold':2, 'Platinum':3,
        'Uneducated':0, 'College':1, 'High School':2, 'Graduate':3, 'Post-Graduate':4, 'Doctorate':5
    }
    for col in dataframe.select_dtypes('object').columns:
        dataframe.loc[:,col] = dataframe[col].map(code)
    
    return dataframe


@app.route("/")
def home():
    return render_template("index.html", title="Accueil")

    
@app.route("/individual", methods=['GET','POST'])
def individual_prediction():
    
    if request.method == 'GET':
        return render_template('index.html', title="Prédiction individuelle")
    
    elif request.method == 'POST':
    
        #For rendering results on HTML GUI
        features_values = [float(x) for x in request.form.values()]
        final_features = pd.DataFrame(features_values, index = features_list).T
        prediction = model.predict(final_features)
        prediction = prediction[0]

        if prediction == 'Existing Customer':
            output = 'Le client restera abonné au service de la carte bancaire'
        else:
            output = 'Le client envisage de mettre fin au service de la carte bancaire'

        return render_template("index.html", title="Prédiction individuelle", prediction_text='{}'.format(output))



@app.route('/group', methods=['GET', 'POST'])
def group_prediction():
    
    if request.method == 'GET':
        return render_template('index.html')

    elif request.method == 'POST':
        results  = []
        
        csv_file = request.form.get('csv_file').split('\n')
        reader = csv.DictReader(csv_file, delimiter=';')
        
        for row in reader:
            results.append(dict(row))
        
        data = pd.DataFrame(results)
        df = data[['CLIENTNUM','Gender','Marital_Status','Income_Category','Card_Category','Months_Inactive_12_mon','Avg_Utilization_Ratio','Total_Relationship_Count','Attrition_Flag']]
        fieldnames = [item for item in df.columns]
        
        df1 = df.T
        
        new_results=[]
        for i in range(df1.shape[1]):
            new_results.append(df1[i].tolist())
        
    
        
        df_new = df.drop('CLIENTNUM', axis=1)
        df_new = df_new.drop('Attrition_Flag', axis=1)        
        df_new =  encodage(df_new)
        prediction = model.predict(df_new)
        
        return render_template('index.html', title="Prédiction de groupe", new_results=new_results, fieldnames=fieldnames, prediction=prediction, len= len)

if __name__ == "__main__":
    app.run(debug=True)