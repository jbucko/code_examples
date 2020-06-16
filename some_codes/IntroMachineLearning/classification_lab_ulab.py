
"""
This code performs classification on both labelled and unlabelled data. First we train NN on labeled data
and predict the labels on unlabeled dataset. Then NN is slightly modified and trained on both datasets to predict
labels on test dataset. Procedure is repeated several times for differently shuffled data. Final label for test instance
is chosen by majority vote
"""
import numpy as np
import pandas as pd
import keras
from keras import initializers
from keras.layers import Dense, Dropout, Activation
from keras import initializers
from numpy import sqrt
from sklearn.decomposition import PCA
from sklearn import preprocessing, utils
from numpy import sqrt
import sklearn.neighbors as ngh
from sklearn.cluster import MiniBatchKMeans as MBKNN
from sklearn.naive_bayes import GaussianNB
from pandas import DataFrame as df
from keras.layers.normalization import BatchNormalization


print("_______________________________________________________")

"""
reading and preparing the data
"""
np.random.seed(52)
train_data_lab= pd.read_hdf("train_labeled.h5", "train")
train_data_ulab= pd.read_hdf("train_unlabeled.h5", "train")
test_data = pd.read_hdf("test.h5", "test")
ytrainL = train_data_lab.y

XtrainL = train_data_lab
XtrainU = train_data_ulab
XtrainL = XtrainL.drop(XtrainL.columns[0], axis = 1)
Xtest = test_data

"""
class statistics for labeled data
"""
u, yc = np.unique(ytrainL, return_counts = True)
ycounts = dict(zip(u,yc))
print('labeled: \n',ycounts)

"""
normalize and transform data
"""
scaler = preprocessing.StandardScaler().fit(pd.concat([XtrainL,XtrainU]))

XtrainL = scaler.transform(XtrainL)
XtrainU = scaler.transform(XtrainU)
Xtest=scaler.transform(Xtest)

XtrainU = pd.DataFrame(XtrainU)
Xtest = pd.DataFrame(Xtest)

no_components = 139 # no. of features per instance


# NNET
def add_layer(no_neurons,no_pred):
  """
  Function adding hidden layer
  Param no_neurons: number of neurons in hidden layer
  Param no_pred: number of neurons in previous layer due to correct ReLU normalization
  Return: 0
  """
  model.add(Dense(no_neurons,
    kernel_initializer = initializers.RandomNormal(mean=0.0,stddev=np.sqrt(2/no_pred),seed=None), activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  return 0


neurons = [1000] # optimize over number of neurons in layer
neurons0 = [1000] 

"""
iterate 14 times to obtain statistics for predicted labels
"""
for i in range(14):
  print("iteration ", i)
  Xtrain = XtrainL
  ytrain = ytrainL
  #shuffle the train data
  idx = np.random.permutation(len(Xtrain))
  Xtrain = Xtrain[idx]
  ytrain = ytrain[idx]
  #split to train and validation set
  Xtrain_val = Xtrain[8100:]
  ytrain_val = ytrain.values[8100:]
  Xtrain = pd.DataFrame(Xtrain[:8100])
  ytrain = ytrain[:8100]

  es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto', baseline=None, restore_best_weights=False) #early stopping
  mc = keras.callbacks.ModelCheckpoint('GNC_PCA.h5',monitor='val_acc',mode='max',verbose=1,save_best_only=True) #save best model

  sgd = keras.optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True) # stochastic gradient descent

  """
  train of labelled data
  """
  for l1 in neurons0:
      for l2 in neurons:

          print("-------------")
          print('l1,l2 = ',l1,',',l2)

          # first hidden layer
          model = keras.Sequential()
          model.add(Dense(l1,
           kernel_initializer = initializers.RandomNormal(mean=0.0,stddev=np.sqrt(2/no_components),seed=None),input_dim=no_components, activation = 'relu'))
          model.add(BatchNormalization())
          model.add(Dropout(0.5))

          # second hidden layer
          if l2 != 0:
              add_layer(l2,l1)
          add_layer(500, l2)
          model.add(Dense(10, activation='softmax'))

          model.compile(optimizer='adam', 
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
          fitted = model.fit(Xtrain, ytrain, epochs=5000,verbose = 2,validation_data=(Xtrain_val, ytrain_val), 
                  shuffle = True, callbacks = [es,mc], batch_size=75) # train the model


  saved_model = keras.models.load_model('GNC_PCA.h5') # load best model

  

  predictions = saved_model.predict(XtrainU) # predict for unlabelled data
  ytrainU = np.argmax(predictions, axis=1) 
  # merge training data
  Xtrain = pd.concat([XtrainU,pd.DataFrame(Xtrain)])
  ytrain = np.hstack([ytrainU, ytrain])


  """
  train on merged dataset
  """
  for l1 in neurons0:
      for l2 in neurons:

          print("-------------")
          print('l1,l2 = ',l1,',',l2)
          model = keras.Sequential()
          model.add(Dense(l1,
           kernel_initializer = initializers.RandomNormal(mean=0.0,stddev=np.sqrt(2/no_components),seed=None),input_dim=no_components, activation = 'relu'))
          model.add(BatchNormalization())
          model.add(Dropout(0.5))

          if l2 != 0:
              add_layer(l2,l1)

          model.add(Dense(10, activation='softmax'))

          model.compile(optimizer='adam', 
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
          fitted = model.fit(Xtrain, ytrain, epochs=5000,verbose = 2,validation_data=(Xtrain_val, ytrain_val), 
                  shuffle = True, callbacks = [es,mc], batch_size=75)


  saved_model = keras.models.load_model('GNC_PCA.h5')
  saved_model.fit(Xtrain_val, ytrain_val, epochs=3,verbose = 2, 
                  shuffle = True, batch_size=75)
  


  predictions = saved_model.predict(Xtest) # predict on test dataset
  pred_df = pd.DataFrame(predictions)
  filename = "predictions"+str(i)+".csv"
  pred_df.to_csv(filename) # save the data
  print("Predictions saved.")
  print("-----------------------------")


""" 
average the results
"""
n=14
predictions_cum = np.zeros(shape=(8000,10))
for i in range(n):
  filename = "predictions"+str(i)+".csv"
  resi = pd.read_csv(filename).values[:,1:]
  #print(resi)
  predictions_cum = np.add(predictions_cum, resi)
  #print(predictions_cum)

predictions_cum = np.divide(predictions_cum, n)
labels = np.argmax(predictions_cum, axis=1) 


#output in required format
ans_file = open('ans_average.csv', 'w')
ans_file.write("Id,y\n")
for i in range(len(labels)):
    ans_file.write(str(i + 30000) + "," + str(labels[i]) + "\n")

#output frequency of labels for sanity check
u, yc = np.unique(labels, return_counts=True)
testcounts = dict(zip(u,yc))
print(testcounts)


