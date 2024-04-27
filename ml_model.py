import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


data=pd.read_csv(r'C:\Users\server\Desktop\Social_Network_Ads.csv')
data=data.drop(['User ID'],axis=1)

le=LabelEncoder()
data['Gender']=le.fit_transform(data['Gender'])

y=data.Purchased
X=data.drop(['Purchased'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=.25)

dt_clf= DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)
y_pred=dt_clf.predict(X_test)

import pickle
with open('model.pkl','wb') as model_file:
    pickle.dump(dt_clf,model_file) 

