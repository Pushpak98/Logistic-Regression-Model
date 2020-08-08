# Logistic-Regression-Model

# Problem

This data set describes operating conditions of a reactor and contains class labels about whether the reactor will operate or fail under those operating conditions. Your job is to construct a logistic regression model to predict the same.
Dataset_Question2.xlsx: The data contains a 1000 X 6 data matrix. The first five columns are the operating conditions of the reactor. The sixth column provides necessary annotation:
● Temperature: 400-700 K
● Pressure: 1-50 bar
● Feed Flow Rate: 50-200 kmol/hr
● Coolant Flow Rate: 1000-3600 L/hr
● Inlet Reactant Concentration: 0.1-0.5 mole fraction
● Test: fail/pass. Whether the reactor will operate or fail under the corresponding operating conditions.
Using the above datasets, make a report on the following: Note: Any assumptions made should be properly mentioned. 
1. Describe the statistics of the data.
2. Partition your data into a training set and a test set. Keep 70% of your data for training and set aside the remaining 30% for testing.
3. Fit a logistic regression model on the training set. Choose an appropriate objective function to quantify classification error. Manually code for the gradient descent procedure used to find optimum model parameters. (Note: You may need to perform multiple initializations to avoid local minima)
4. Evaluate the performance of above model on your test data. Report the confusion matrix and the F1 Score.

# Solution

Following are the steps required to implement Logistic Regression -
1. Collecting Data
2. Analyzing and Cleaning the Data
3. Train and Test
4. Accuracy check

__Step 1 – Collecting Data__ : This step involves importing the libraries that are used to collect the data and then
taking it forward. Look at the code q2_.py
1. To work with data frames, we have to import Pandas library.
2. To perform any numerical operations on it we need to import the Numpy library.
3. We have to import train_test_split from the package called sklearn and model selection is a sub package
under sklearn(scikit-learn).
4. After this we import LogisticRegression from scikit learn linear_model and for performance metrics we
need to import accuracy score and confusion matrix from sklearn metrics.
5. After importing the data it is stored in data_reactor DataFrame. We have also created a copy of the
original data in a new DataFrame named as data .

__Step 2 – Analyzing and Cleaning the Data__ : In this step we first analyze the data that is we do exploratory data
analysis in which we have 3 steps 

__*Getting to know the data*__ – By checking variables’ data type, we have noticed that in each of the columns there
are 1000 non null float values except in the last column in which it has a non null object as expected because we
know that there are categories under this variable so it has been read as object.

__Data preprocessing – (To check for missing values)__ : using isnull function we see whether there are any missing
values in the data or not. This isnull returns Boolean value that is True (indicates missing data) and False
(indicates that there are no missing values in a particular cell). But this is a 1000 X 6 matrix it is difficult to skim
through the whole data matrix. So if we take sum we can clearly get how many cells are missing under each
variable. By looking at the output we can say that there are no missing values.

For checking other symbols and unique classes we can use value_counts for each column but we can see a lot of
values in each column which doesn’t give any insight about the data, instead we performed it only for Test
column which gave us Pass = 585 and Fail = 415 it means there are no special symbols in this column. So if there
exists a special symbol we can view them using a unique function. So we went through checking all the variables
using a unique function but it is difficult to skim through all the points. So finally we concluded that there are no
other special symbols in the dataset.

__Cross tables and Data Visualization (Statistics of the Data)__ – Last step is to look into the relationship between
independent variables using the correlation measure. So we got the pair wise correlation function as
We know correlation values lie from -1 to 1 and if they are closer to 1 we say that there is a strong relationship
between two variables but in our case none of the values are nearer to 1which represents none of them are
correlated to each other.

</img><img src="https://user-images.githubusercontent.com/55409875/89710644-8820c600-d9a2-11ea-8875-7573a0b404c7.PNG" width="15%"></img>

we need to convert all the string values to numerical values for which we need to re-index the categories to
numerical values that are Pass to 1 and Fail to 0 so that we can work with any machine learning algorithms. Now
we need to select the features where we need to divide the given columns into two types : one having
independent and other having dependent variables. For this we store the column names and separate input
names from the data. Here we took y as dependent and x as in dependent variable and store the values from
input features, output values in y.

__Step 3 – Train and Test set__
In this we have to divide the data into train sets and test sets so that we can build the model on the train set and
deserve some part of the data to test the model on. The input parameters for the train test split would be
x(input values) and y(output values) and as mentioned in the question 30% for testing the data and 70% for
training we used test_size = 0.3 which represents the same. Under train we have two sets of data which has only
the input variables and the other will have only the output variable similarly under test set we have input
variables separately and the output variable separately. We can observe the dimensions in the variable explorer
as well. Now we construct a logistic regression classifier instance in which we can fit the model using the fit
function. We can also get some attributes from classification models like coefficients, intercept etc. Now we can
predict the model on the data frame using the predict function.

__Step 4 – Accuracy check__
We can evaluate the model using the confusion matrix which is used to evaluate performance of a classification
model. This tells us about no of correct predictions and no of incorrect predictions. We have got the confusion
matrix and accuracy score as 87.67 %. In the confusion matrix if the actual class is Fail that is 0 then our model
has predicted 104 observations but being Fail as the actual class the model has predicted 23 observations as
Pass. Similarly, the model has predicted 159 observations as Pass and 14 observations as Fail. So it means there
are many misclassified values which were found to be 37 observations.
__Confusion matrix -__
</img><img src="https://user-images.githubusercontent.com/55409875/89710648-91119780-d9a2-11ea-8e37-34b013201608.PNG" width="15%"></img>

__Questions -__
1. Statistics of the data
We can get the statistics of the data using describe function –
 </img><img src="https://user-images.githubusercontent.com/55409875/89710649-9242c480-d9a2-11ea-969e-a0f12ceb37e2.PNG" width="15%"></img>

We have also got a pair wise correlation function that is attached in the 2nd step.
2. This part was done earlier in step 3 – Train and Test set.
3. For this we are going to use Newton’s method to solve the Logistic Regression model and to code for
gradient descent as well. The only assumption we made is we chose the learning rate to be 0.2 initially. However,
it can be changed and used in the code (q2_Newtons_method.py) and predict the accuracy accordingly. In the first
step we import all the necessary libraries for the model and we define absolute function and sigma function. We
also define two more functions that are R and b (A,W,n) as our arguments for the gradient descent part in which
we have to minimize the error (loss function i.e; objective function). In R we return m of all zeros and in b we
have a matrix.
In another function “testing” we have test_data and w as our arguments in which we are going to calculate
accuracy score and confusion matrix. We considered the threshold to be 0.5 and the values more than this are
noted as 1 that is Pass and less than that noted as 0 that is Fail.
Note
1 st assumption – learn_rate = 0.2(we have to keep changing this and to check for the best accuracy score)
2 nd assumption – Threshold value = 0.5 (generally)
4. At one particular time,
Performance of the above model says – Precision was observed to be 0.946 and
Recall equals to 0.961
 </img><img src="https://user-images.githubusercontent.com/55409875/89710654-9cfd5980-d9a2-11ea-876f-911fde9860b4.PNG" width="15%"></img>

F1 Score is the harmonic mean of Precision and Recall that we evaluated in Performance of the model.
Accuracy score = 0.96
Accuracy = 96%
Confusion matrix – output[0] is accuracy and output[1] is Confusion matrix

</img><img src="https://user-images.githubusercontent.com/55409875/89710662-a2f33a80-d9a2-11ea-9114-b20632c1ade2.PNG" width="15%"></img> <img src="https://user-images.githubusercontent.com/55409875/89710664-a5ee2b00-d9a2-11ea-9e7a-ab686f3b0b06.PNG" width="15%"></img> 
