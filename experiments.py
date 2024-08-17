import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle


data=pd.read_csv('Churn_modelling.csv')
print(data.head())

#pre processing the data

#drop irrelevant columns

columns=data.columns
for i in columns:
    print(i,type(i))
print('------------------------------------')
# in this pre processing we are removing the RowNumber,CustomerId,Surname

data=data.drop(['RowNumber','CustomerId','Surname'],axis=1)
print(data.columns)
print(data.head(1))
print('------------------------------------')

# in this step we are labeling the categorical variables

# encode categorical variables
label_encoder_gender=LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])
print(data.head(2))
print('--------------------------------------')

from sklearn.preprocessing import  OneHotEncoder
onehot_encoder_geo=OneHotEncoder()
geo_encoder=onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
print(geo_encoder)

print('----------------------------------')

print(onehot_encoder_geo.get_feature_names_out(['Geography']))

print(data['Geography'])


geo_encoded_df=pd.DataFrame(geo_encoder,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
print(geo_encoded_df)
print('-------------------------------------')
## Combine one hot encoder columns with the original data
data=pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)
print(data.head())
print('-------------------------------------')


## save the encoders and scaler

with open("label_encoder_gender.pkl",'wb') as file:
    pickle.dump(label_encoder_gender,file)
with open('onehot_encoder_geo.pkl','wb') as file:
    pickle.dump(onehot_encoder_geo,file)

#dividing the data into independent and dependent
X=data.drop('Exited',axis=1)
y=data['Exited']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=35)

#scaling the features

scaler=StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

print(X_train)

with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)


####ANN implementation

import  tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import datetime

print('-------------------------------')
print((X_train.shape[1],))
print('-------------------------------')

#building ANN model
model=Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])

print(model.summary())
print('---------------------------------')
import  tensorflow
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)
tensorflow.keras.losses.BinaryCrossentropy()


#complie the model
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])


#setup the tensor board

from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

log_dir='logs/fit/'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

#setup early stopping

early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

### Train the model
history=model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)

model.save('model.h5')

