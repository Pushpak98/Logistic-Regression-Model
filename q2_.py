
"  # importing libraries"
import pandas as pd
import numpy as np
import seaborn as sbs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


"  #importing data"
data_reactor = pd.read_excel("Dataset_Question2.xlsx")
#data_reactor = pd.read_csv("Dataset_Question2.csv")
#print(data_reactor)

" #copy of original data"
data = data_reactor.copy()
#print(data)
print(data.info())


"#checking for Nan Values"
print(data.isnull())
print(data.isnull().sum())
print(data.isnull().sum().sum())
        #conclusion - No Nan values in the dataset

"#checking for other symbols and unique classes"
print(data["Test"].value_counts())

T = np.unique(data["Temperature"])
print(T)
P = np.unique(data["Pressure"])
print(P)
FFR = np.unique(data["Feed Flow rate"])
print(FFR)
CFR = np.unique(data["Coolant Flow rate"])
print(CFR)
IRC = np.unique(data["Inlet reactant concentration"])
print(IRC)
        #conclusion - No special symbols as well


" #statistics of data"
summary_num = data.describe()
print(summary_num)
summary_cate = data.describe(include = "O")
print(summary_cate)
correlation = data.corr()
print(correlation)

"# reindexing the TEST names to 0,1"
data["Test"] = data["Test"].map({"Fail" : 0,"Pass": 1})
print(data)
"""
new_data =pd.get_dummies(data, drop_first = True) 
print(new_data)
"""

"#storing the col names and separating input names from data"
col_list = list(data.columns)
print(col_list)

features = list(set(col_list)-set(["Test"]))
print(features) 

"#storing the values from input features, output values in y"
x = data[features].values
y = data["Test"].values
print(x)
print(y)

"#splitting into train and test "
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)
logistic = LogisticRegression()
"#fitting the values for x and y"
logistic.fit(x_train,y_train) 
logistic.coef_
logistic.intercept_
logistic.class_weight

#prob = logistic.predict_proba(x_test)
#print(prob)
"#prediction from test data"
prediction = logistic.predict(x_test)
print(prediction)

"#confusion matrix"
confusion_matrix = confusion_matrix(y_test, prediction)
print(confusion_matrix)

"#accuracy score"
accuracy_score = accuracy_score(y_test, prediction)
print(accuracy_score)

"#classification report"
from sklearn.metrics import classification_report 
print(classification_report(y_test,prediction))

"#misclassified values from prediction"
print("no of misclassified : %d" %(y_test != prediction).sum())

"#Mean absolute error"
from sklearn.metrics import mean_absolute_error 
MAE = mean_absolute_error(y_pred=prediction, y_true = y_test)
print(MAE)
"#conc - so there are 37 missclassified samples "

#---------------------------------------------------------------------------------------------------------