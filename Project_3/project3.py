# -*- coding: utf-8 -*-
"""
Neural Networks to predict protein activation
idate
Created by Sebastian B. 2021_05_12
"""
import time
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0,1.2,3
import pandas as pd
import tqdm
import numpy as np
import pickle
import re
import tensorflow as tf
# keras imports
import keras
from keras.layers import Dense, Dropout, BatchNormalization, AlphaDropout
from keras import initializers, Sequential, regularizers
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import Callback,ModelCheckpoint, EarlyStopping
# sklearn imports
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight

def get_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def get_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def get_f1(y_true, y_pred):
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def myLogger(logLevel):
    path = os.path.dirname(__file__)
    logger = logging.getLogger(__name__)
    logger.setLevel(logLevel)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    file_handler_e = logging.FileHandler(f'{path}/log.log')
    file_handler_e.setLevel(logging.INFO)
    file_handler_e.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler_e)
    logger.addHandler(stream_handler)
    return logger

class Model_Connected():
    def __init__(self, hiddensize, seed):
        self.hiddensize = hiddensize
        self.seed = seed

        self.model = Sequential()
        self.model.add(Dense(HIDDENSIZE, input_dim = 80, activation='relu', kernel_initializer= initializers.lecun_normal(seed=SEED)))
        self.model.add(Dropout(rate=0.1))
        self.model.add(BatchNormalization())
        self.model.add(Dense(107, input_dim = HIDDENSIZE, activation='relu', kernel_initializer= initializers.lecun_normal(seed=SEED)))
        self.model.add(Dropout(rate = 0.1))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1, input_dim = 107, activation='sigmoid'))

        self.opt = keras.optimizers.Adam()  # alternative: opt = SGD(lr=0.01, momentum=0.9)

    def compile(self):
        self.model.compile(optimizer=self.opt, loss="binary_crossentropy", metrics=['accuracy', get_f1])  # f1 score should be off the shelf with sklearn
        # self.model.compile(optimizer=self.opt, loss='mean_squared_error', metrics=['accuracy', get_f1])
        return self.model

if __name__ == "__main__":
    # defining global variables
    NEPOCHS = 200
    BATCHSIZE = 500
    VALIDATIONSPLIT = 0.2
    HIDDENSIZE = 500
    SEED = 29
    VERBOSE = True
    LOGLEVEL = logging.INFO

    # define logger
    logger = myLogger(LOGLEVEL)

    """#### Load data & data inspection"""
    dat_train = pd.read_csv("./data/train.csv")
    dat_test = pd.read_csv("./data/test.csv")
    logger.info(f"Pecentage of active values: {np.round(100*np.mean(dat_train['Active']),2)}")  # check class balance on activation

    """#### Pre-process data"""
    strToFeat = lambda X: np.array([list(s) for s in X])  #converts the protein sting to seperate features
    X, y = strToFeat(dat_train['Sequence']), dat_train['Active']
    X_test = strToFeat(dat_test['Sequence'])
    X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=1-VALIDATIONSPLIT,stratify=y)
    logger.debug(X[:10])
    logger.debug(X_train[:10])

    """ binary/one-hot encoding"""  #TODO: Check if it would be better to eliminate one hot encoding
    onehot_encoder = OneHotEncoder(sparse=True)
    X_train_hot = onehot_encoder.fit_transform(X_train).toarray()
    X_val_hot = onehot_encoder.fit_transform(X_val).toarray()
    X_test_hot = onehot_encoder.transform(X_test).toarray()
    logger.debug(f'Shape of X_train {X_train.shape}')
    logger.debug(f'Shape of X_train_hot {X_train_hot.shape}')

    """ Create Model"""
    model_inst = Model_Connected(HIDDENSIZE, SEED)
    model = model_inst.compile()
    logger.info(model.summary())

    """Train Model"""
    best_fold = -1
    best_score = 0
    best_model = None

    # train
    weight = {1:y_train.shape[0]/np.sum(y_train), 0:1.0}
    logger.debug("Model compile start...")
    model = model_inst.compile()
    logger.debug("Model train start ...")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_get_f1',  patience=40, mode="max", min_delta=0.01)
    model.fit(X_train_hot, y_train, validation_data = (X_val_hot, y_val), epochs = NEPOCHS, batch_size=BATCHSIZE, verbose = VERBOSE, class_weight = weight, callbacks=[callback])
    # predictions
    logger.debug("Checking validation score...")
    y_val_pred= (model.predict(X_val_hot) > 0.5).astype("int32")
    val_f1 = metrics.f1_score(y_val,y_val_pred)
    logger.info(f"F1 validation score is : {val_f1}")

    """#### Prediction on test data"""
    y_test = (model.predict(X_test_hot) > 0.5).astype("int32")
    logger.info(f"The percentage of predicted active proteins is: {np.round(100*np.sum(y_test)/y_test.shape[0])}")
    y_test_df = pd.DataFrame(y_test)

    """ Save results"""
    timestamp = int(time.time())
    y_test_df.to_csv(f"./results/results_{timestamp}.csv", index=False, header=False)
    logger.info("Results successfully saved, program finished.")

