# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 2 hidden layer with relu activation function,one input layer and one output layer . Now we will fit our dataset and then predict the value.


## Neural Network Model :

![image](https://github.com/vishnudorigundla/basic-nn-model/assets/94175324/55452805-e3e1-49ce-8353-4596daafd396)

## DESIGN STEPS :

### STEP 1: 
Loading the dataset

### STEP 2:
Split the dataset into training and testing

### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:
Build the Neural Network Model and compile the model.

### STEP 5:
Train the model with the training data.

### STEP 6:
Plot the performance plot

### STEP 7:
Evaluate the model with the testing data.

## PROGRAM :
```
Developed By : D.vishnu vardhan reddy
Reference Number : 212221230023
```
### Importing Required Packages :
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```
### Authentication and Creating DataFrame From DataSheet :
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('StudentsData').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df
```
### Assigning X and Y values :
```
X = df[['INPUT']].values
Y = df[['OUTPUT']].values
```
### Normalizing the data :
```
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
```
### Creating and Training the model :
```
model = Sequential([
    Dense(5,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss = 'mse')
model.fit(X_train1,y_train,epochs=2200)
```
### Plot the loss :
```
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
```
### Evaluate the Model :
```
X_test1 = Scaler.transform(X_test)
model.evaluate(X_test1,y_test)
model.evaluate(X_test1,y_test)
```
### Prediction for a value :
```
X_n1 = [[20]]
X_n1_1 value = Scaler.transform(X_n1)
model.predict(X_n1_1 value)
```


## Dataset Information :

![image](https://github.com/vishnudorigundla/basic-nn-model/assets/94175324/95480c13-bcb3-45bf-a130-418ad71cae82)
![image](https://github.com/vishnudorigundla/basic-nn-model/assets/94175324/35eb7809-26fe-4107-a111-f61c7cbf66c0)


## OUTPUT :

### Training Loss Vs Iteration Plot

![image](https://github.com/vishnudorigundla/basic-nn-model/assets/94175324/a4e61011-0392-4865-bf8f-1aa8baf93cc6)


### Test Data Root Mean Squared Error :
![image](https://github.com/vishnudorigundla/basic-nn-model/assets/94175324/91940c5d-f88f-4e97-ac3f-2f993fd3c3b0)


### New Sample Data Prediction

![image](https://github.com/vishnudorigundla/basic-nn-model/assets/94175324/7927fc6a-0ed5-451d-a210-09112af6795f)


## RESULT :
Thus the neural network regression model for the given dataset is executed successfully.
