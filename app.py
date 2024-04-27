from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle


app=Flask(__name__)
data=pd.read_csv('Social_Network_Ads.csv')
data=data.drop(['User ID'],axis=1)

with open('model.pkl','rb') as model_file:
    model=pickle.load(model_file)

@app.route('/')

def home():
 return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Age=int(request.form['age'])
    
    Gender=(request.form['Gender'])
    Salary=int(request.form['salary'])
    if Gender=="male":
        Gender=0
    else:
        Gender=1

    features=np.array([[Gender,Age,Salary]])
    prediction=model.predict(features)

    if  prediction == 0:
        result= " Not purchased"
    else:
        result=' Purchased'
    print(result)
    
    return render_template('index.html',res=result)

if __name__ == '__main__':
       app.run(debug=True)
