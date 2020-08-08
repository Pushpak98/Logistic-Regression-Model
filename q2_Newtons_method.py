import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from random import shuffle
from pylab import *
from random import randint
import pandas as pd
from matplotlib import colors

def abso(W):
    temp = np.sum(W**2,axis = 0)
    return ((temp[0])**0.5)

def sigma(a):
    return (1/(1+exp(-a)))

def R(A,W,n): 
    m = np.zeros((n,n),dtype = float)
    for i in range(n):
        temp = sigma(np.matmul(W.T,A[i])[0])
        m[i][i] = temp*(1-temp)
    return m;

def b(A,W,n):
    mat = np.zeros((n,1), dtype = float)
    for i in range(n):
        mat[i][0] = sigma(np.matmul(W.T,A[i])[0])
    return mat


def testing(test_data,W):
    X = np.zeros((6,1),dtype = float)
    accuracy = 0.0
    output = 0
    CONF_MATRIX=array([[0,0],[0,0]])
    for sample in test_data:
        X[0][0] = 1
        X[1][0] = float(sample[0])
        X[2][0] = float(sample[1])
        X[3][0] = float(sample[2])
        X[4][0] = float(sample[3])
        X[5][0] = float(sample[4])
        if (sigma(np.matmul(W.T,X)[0][0]) < 0.5):
            output = 0;
        else:
            output = 1;
        if (output == int(sample[5])):
            accuracy=accuracy+1
        
        CONF_MATRIX[output,int(sample[5])]+=1  
    return (accuracy*100/len(test_data),CONF_MATRIX)

    #print(CONF_MATRIX[0,0])

learn_rate = 0.2

data1 = pd.read_excel("Dataset_Question2.xlsx").to_numpy() 
data = list(data1)[1:]
for i in data:
    if (i[5] == 'Fail'):
        i[5] = 0
    else:
        i[5] = 1
shuffle(data)
n = len(data)
n_train = int(0.7*n)
n_test = n - n_train
train_data = data[:n_train]
test_data = data[n_train:]

# A matrix
A = np.zeros((n_train,6),dtype = float)
for i in range(n_train):
    A[i][0] = 1
    A[i][1] = float(train_data[i][0])
    A[i][2] = float(train_data[i][1])
    A[i][3] = float(train_data[i][2])
    A[i][4] = float(train_data[i][3])
    A[i][5] = float(train_data[i][4])

# #regularization
# for i in range(1,6):
#     A[:,i] = (A[:,i]-min(A[:,i]))/(max(A[:,i])-min(A[:,i]))

# Y matrix
Y = np.zeros((n_train,1),dtype = float)
for i in range(n_train):
    Y[i][0] = float(train_data[i][5])

W = np.random.rand(6,1)
for i in range(0,6):
    W[i][0] = W[i][0]/max(A[:,i])


for i in range(100):
    W_prev = np.copy(W)
    W = W - learn_rate*(np.matmul(np.linalg.inv(np.matmul(np.matmul(A.T,R(A,W,n_train)),A)),np.matmul(A.T,b(A,W,n_train)-Y)))
    if (abso(W-W_prev) < 0.0005*(abso(W))):
            break;

print("No of iteration to converge = ",i)
train_out = testing(train_data,W)
print("accuracy of train_data = " , train_out[0])
print("confusion matrix of train_data = " , train_out[1])
output = testing(test_data,W)
print("accuracy of test_data = " , output[0])
print("confusion matrix of test_data = " , output[1])

#confusion matrix for best model
a = output[1]
row_sums = sum(a,axis=1)
col_sums = sum(a,axis=0)
CONF_MAT = array([[a[0][0],a[0][1],row_sums[0]],[a[1][0],a[1][1],row_sums[1]],[col_sums[0],col_sums[1],0]])

CONF_MAT[2][2] = row_sums[0]+row_sums[1]

import seaborn as sns; sns.set()
sns.set(font_scale = 1.0)
df_cm = pd.DataFrame(CONF_MAT,index = ['Class-0','Class-1',' '],columns = ['Class-0','Class-1',' '])
sns.heatmap(df_cm,annot=True,fmt = 'd',cbar = False)
plt.title('Confusion Matrix on Test_data',size = 15)
plt.xlabel('True label',size = 15,color = 'red')
plt.ylabel('Predicted label',size = 15,color = 'red');
plt.show()


"#Evaluation of Performance and F1 score "

matrix = output[1]
print("confusion matrix on test data", matrix)


Precision = matrix[0,0] / (matrix[0,0] + matrix[0,1])
print(Precision)
Recall = matrix[0,0] / (matrix[0,0] + matrix[1,0])
print(Recall)
F1_score = (2 * Precision * Recall) / (Precision + Recall)
print(F1_score)


#------------------------------------------------------------------------------------

