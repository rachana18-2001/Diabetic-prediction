from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def output(request):
    return render(request,'output.html')

def result(request):
    data = pd.read_csv(r'C:\Users\Sony\Desktop\mini project\DiabeticsPrediction\DiabeticsPrediction\diabetes.csv')
    
    X = data.drop('Outcome', axis=1)#outcome column is dropped
    y= data['Outcome']#it contains only outcome column
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    model = LogisticRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(prediction, y_test)
    
   # clf = SVC(kernel='linear')
    #clf.fit(X_train, y_train)
   # y_pred = clf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    prediction = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    result1 = ""
    if prediction==[1]:
        result1="Positive"
    else:
        result1="Negative"
        
    return render(request, "output.html",{"result2":result1})
