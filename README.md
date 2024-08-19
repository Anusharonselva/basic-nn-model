# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
A neural network can be used to solve a problem by Data collection and Preprocessing,choosing a appropriate neural network architecture ,Train the neural network using the collected and preprocessed data,Assess the performance of the trained model using evaluation metrics,Depending on the performance of the model, you might need to fine-tune hyperparameters or adjust the architecture to achieve better results,Once you're satisfied with the model's performance, you can deploy it to production where it can be used to make predictions on new, unseen data

For the problem statement we have dealt with , we have developed a neural network with three hidden layers. First hidden layer consists of 4 neurons ,second hidden layer with 8 neurons , third layer with 5 neurons . The input and output layer contain 1 neuron . The Activation Function used is 'relu'.
## Neural Network Model

![Screenshot 2024-08-19 143104](https://github.com/user-attachments/assets/e6744192-afc7-49f7-a7ba-a075465279fc)

## DESIGN STEPS

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

## PROGRAM
### Name: S.ANUSHARON
### Register Number:212222240010
```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds,_=default()
gc = gspread.authorize(creds)

worksheet = gc.open('Mark').sheet1
data = worksheet.get_all_values()

df=pd.DataFrame(data[1:],columns=data[0])
df= df.astype({'X':'int'})
df= df.astype({'Y':'int'})
df.head()

x=df[['X']].values
y=df[['Y']].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
Scaler=MinMaxScaler()

Scaler.fit(x_train)

x_train1=Scaler.transform(x_train)

ai_brain=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])

ai_brain.compile(optimizer ='rmsprop',loss='mse')

ai_brain.fit(x_train1,y_train,epochs=2000)

loss_df=pd.DataFrame(ai_brain.history.history)

loss_df.plot()

x_test1=Scaler.transform(x_test)

ai_brain.evaluate(x_test,y_test)

x_n1=[[8]]

x_n1_1=Scaler.transform(x_n1)

ai_brain.predict(x_n1_1)

```
## Dataset Information

Include screenshot of the dataset
![Screenshot 2024-08-19 142246](https://github.com/user-attachments/assets/f6e2b544-eb6e-4f5a-8fad-a10a086eec19)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-08-19 143159](https://github.com/user-attachments/assets/80b0cfc8-9ff0-480e-8050-45726e5c980c)


### Test Data Root Mean Squared Error

![Screenshot 2024-08-19 143244](https://github.com/user-attachments/assets/903532d0-ef0d-45ef-a25a-0d0a1c6098b0)


### New Sample Data Prediction

![Screenshot 2024-08-19 143253](https://github.com/user-attachments/assets/1c68ee82-e3bd-49ce-a739-b3eabcad49a9)


## RESULT
Thus a neural network regression model is developed for the created dataset.
