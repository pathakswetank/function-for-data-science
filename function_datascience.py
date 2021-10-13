 def name_your_function( parameters ):
   "function_docstring"
   function_suite
   return [expression]


import pandas as pd
load_data = pd.read_csv("xyz.csv")

import os
import tarfile
import urllib
DOWNLOAD_ROOT = "https://yourrepo.....//..../"
XYZ_PATH = os.path.join("datasets", "xyz")
XYZ_URL = DOWNLOAD_ROOT + "datasets/xyz/xyz.tgz"
def fetch_housing_data(xyz_url=XYZ_URL, xyz_path=XYZ_PATH):
   os.makedirs(xyz_path, exist_ok=True)
   tgz_path = os.path.join(xyz_path, "xyz.tgz")
   urllib.request.urlretrieve(xyz_url, tgz_path)
   xyz_tgz = tarfile.open(tgz_path)
   xyz_tgz.extractall(path=xyz_path)
   xyz_tgz.close()



import pandas as pd

# this function will return the data
def load_data(xyz_path = XYZ_PATH):
   csv_path = os.path.join(xyz_path, "xyz.csv")
  
   return pd.read_csv(csv_path)


def exploratory_data_analysis(data):
   # rows and column
   rows = data.shape[0]
   columns = data.shape[1]
   # data type
   data_type = [data.dtypes for column in data.columns]
   #Top five rows
   top = data.head()
   #Last five rows
   last = data.tail()
   #descriptive stats
   describe = data.describe()
   # Missing Value
   missing = data.isnull().sum()
   # duplicate values
   dups = sum(data.duplicated())
   # Pearson Correlation
   plt.subplots(figsize=(25,10))
   correlation = sns.heatmap(data.corr(), annot = True)
   plt.subplots(figsize=(25,10))
   #outlier detection
   outlier = data.boxplot();
   #bivariate analysis
   bivariate = sns.pairplot(data)
   #creating a dictionary
   eda = {'Total Rows':rows,
          'Total Columns': columns,
          'Data Type': data_type,
          'Top_five' : top,
          'Last_five' : last,
          'Statistical_Summary': describe,
          'Missing_Value': missing,
          'Duplicate Value': dups,
          'Correlation': correlation,
          'Outlier Detection': outlier,
          'Bivariate Analysis': bivariate}
   
   return eda

def remove_outlier(col):
   sorted(col)
   Q1,Q3=np.percentile(col,[25,75])
   IQR=Q3-Q1
   lower_range= Q1-(1.5 * IQR)
   upper_range= Q3+(1.5 * IQR)
   return lower_range, upper_range



#machine learning

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

lr = LinearRegression()
dr = DecisionTreeRegressor()
rr = RandomForestRegressor()

#creating a function model
def model(X,y,model_1,model_2,model_3):
  
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size = 0.33, random_state = 42)
   model_1.fit(X_train, y_train)
   model_2.fit(X_train, y_train)
   model_3.fit(X_train, y_train)
   model_1_accuracy = model_1.score(X_train, y_train)
   model_2_accuracy = model_2.score(X_train, y_train)
   model_3_accuracy = model_3.score(X_train, y_train)
  
   overall = pd.DataFrame({"Model 1 Accuracy" : model_1_accuracy,
              "Model 2 Accuracy" : model_2_accuracy,
              "Model 3 Accuracy" : model_3_accuracy}, index =[0])
   return overall.transpose()
model(X,y,lr,dr,rr)





from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

svc = LinearSVC()
log = LogisticRegression(random_state=0)
rf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
#creating function model
def model(X,y,model_1,model_2,model_3):
   
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size = 0.33, random_state = 42)
   model_1.fit(X_train, y_train)
   model_2.fit(X_train, y_train)
   model_3.fit(X_train, y_train)
   model_1_accuracy = model_1.score(X_train, y_train)
   model_2_accuracy = model_2.score(X_train, y_train)
   model_3_accuracy = model_3.score(X_train, y_train)
  
   model_1_predict_train = model_1.predict(X_train)
   model_1_predict_test = model_1.predict(X_test)
  
   model_2_predict_train = model_2.predict(X_train)
   model_2_predict_test = model_2.predict(X_test)
  
   model_3_predict_train = model_3.predict(X_train)
   model_3_predict_test = model_3.predict(X_test)
  
   model_1_clf_train = classification_report(y_train,model_1_predict_train,output_dict=True)
   df=pd.DataFrame(model_1_clf_train).transpose()
   model1_train_precision=round(df.loc["1"][0],2)
   model1_train_recall=round(df.loc["1"][1],2)
   model1_train_f1=round(df.loc["1"][2],2)
model_1_clf_test = classification_report(y_test,model_1_predict_test,output_dict=True)
   df=pd.DataFrame(model_1_clf_test).transpose()
   model1_test_precision=round(df.loc["1"][0],2)
   model1_test_recall=round(df.loc["1"][1],2)
   model1_test_f1=round(df.loc["1"][2],2)
model_2_clf_train = classification_report(y_train,model_2_predict_train,output_dict=True)
   df=pd.DataFrame(model_2_clf_train).transpose()
   model2_train_precision=round(df.loc["1"][0],2)
   model2_train_recall=round(df.loc["1"][1],2)
   model2_train_f1=round(df.loc["1"][2],2)
model_2_clf_test = classification_report(y_test,model_2_predict_test,output_dict=True)
   df=pd.DataFrame(model_2_clf_test).transpose()
   model2_test_precision=round(df.loc["1"][0],2)
   model2_test_recall=round(df.loc["1"][1],2)
   model2_test_f1=round(df.loc["1"][2],2)
model_3_clf_train = classification_report(y_train,model_3_predict_train,output_dict=True)
   df=pd.DataFrame(model_3_clf_train).transpose()
   model3_train_precision=round(df.loc["1"][0],2)
   model3_train_recall=round(df.loc["1"][1],2)
   model3_train_f1=round(df.loc["1"][2],2)
model_3_clf_test = classification_report(y_test,model_3_predict_test,output_dict=True)
   df=pd.DataFrame(model_3_clf_test).transpose()
   model3_test_precision=round(df.loc["1"][0],2)
   model3_test_recall=round(df.loc["1"][1],2)
   model3_test_f1=round(df.loc["1"][2],2)
overall = pd.DataFrame({"Model 1 Accuracy" : model_1_accuracy,
                           "Model 1 Train Data Precision" : model1_train_precision,
                           "Model 1 Train Data Recall" : model1_train_recall,
                           "Model 1 Train Data F1 Score" : model1_train_f1,
                           "Model 1 Test Data Precision" : model1_test_precision,
                           "Model 1 Test Data Recall" : model1_test_recall,
                           "Model 1 Test Data F1 Score" : model1_test_f1,
                           "Model 2 Accuracy" : model_2_accuracy,
                           "Model 2 Train Data Precision" : model2_train_precision,
                           "Model 2 Train Data Recall" : model2_train_recall,
                           "Model 2 Train Data F1 Score" : model2_train_f1,
                           "Model 2 Test Data Precision" : model2_test_precision,
                           "Model 2 Test Data Recall" : model2_test_recall,
                           "Model 2 Test Data F1 Score" : model2_test_f1,
                           "Model 3 Accuracy" : model_3_accuracy,
                           "Model 3 Train Data Precision" : model3_train_precision,
                           "Model 3 Train Data Recall" : model3_train_recall,
                           "Model 3 Train Data F1 Score" : model3_train_f1,
                           "Model 3 Test Data Precision" : model3_test_precision,
                           "Model 3 Test Data Recall" : model3_test_recall,
                           "Model 3 Test Data F1 Score" : model3_test_f1}, index = [0])
             
                     
   return overall.transpose()
model(X,y,svc,log,rf)






#passing the features in function
def predict_business(var1, var2, var3,var4, var5, var6, var7):
   a=np.zeros(len(X.columns))
   a[0] = var1
   a[1] = var2
   a[2] = var3 
   a[3] = var4
   a[4] = var5
   a[5] = var6
   a[6] = var7
  
   return (model.predict([a]))


import pickle #importing pickle to load the model
import json #import json to load columns
import numpy as np

#creating a global variables
__data_columns = None
__model = None

#creating a function to pass the features
def predict_business(var1, var2, var3,var4, var5, var6, var7):
   a=np.zeros(len(X.columns))
   a[0] = var1
   a[1] = var2
   a[2] = var3 
   a[3] = var4
   a[4] = var5
   a[5] = var6
   a[6] = var7
   return (__model.predict([a]))

def load_saved_artifacts():
   print("loading saved artifacts...start")
   global  __data_columns
   global __locations
with open("columns.json", "r") as f:
       __data_columns = json.load(f)['data_columns']
     
   global __model
   if __model is None:
       with open('your_file.pickle', 'rb') as f:
           __model = pickle.load(f)
   print("loading saved artifacts...done")

def get_data_columns():
   return __data_columns

if __name__ == '__main__':
   load_saved_artifacts()
   print(predict_business(12,12,12,12,21,12,40))


#The following code block shows the (app.py) application’s full source code:
from flask import Flask, request, jsonify, render_template
import util
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('demo.html')

@app.route('/predict_business', methods=['GET', 'POST'])
def predict_business():
  
   var1 = int(request.form["var1"])
   var2= int(request.form["var2"])
   var3= int(request.form["var3"])
   var4= int(request.form["var4"])
   var5= int(request.form["var5"])
   var6= int(request.form["var6"])
   var7= int(request.form["var7"])
  
   response = util.predict_business(var1,var2,var3,var4,
               var5, var6, var7)
  
  
   if response == 1:
       prediction = {'As per the business attributes you should ' : "invest"}      
   elif response == 0:
       prediction =  {"As per the business attributes you should ' : "not invest"}
   else:
       prediction ={'As per the business attributes you should ' : "research"}
      
    
   return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
   print("Starting Python Flask Server For Win Prediction...")
   util.load_saved_artifacts()
   app.run()
