# -*- coding: utf-8 -*-
"""
Neural Networks to predict protein activation

### Improvement ideas
* Look at one hot key encoding, are we dropping one of the 21 features? do we need to do so?
* One vector with four 1s, we might be loosing information
* Class weights
* Number of epochs
* batch size
* Neural Network (number of layers, where to put drop out layer, activations)
* Optimizer, loss function for f1
"""

import pandas as pd
import numpy as np
import tqdm

import pickle
# %tensorflow_version 2.x
import tensorflow as tf
# from tensorflow.python.util import deprec
# deprecation._PRINT_DEPRECATION_WARNINGS = False

import keras
import keras.backend as K
from keras import Sequential

from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.layers import AlphaDropout

from keras import regularizers
from keras.optimizers import SGD
from keras.callbacks import Callback,ModelCheckpoint, EarlyStopping
from keras import initializers

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder

from keras import backend as K
# import keras.backend.tensorflow_backend as tfback

from tensorflow.python.client import device_lib
import os

import re

# def _get_available_gpus():

    # if tfback._LOCAL_DEVICES is None:
    #     devices = tf.config.list_logical_devices()
    #     tfback._LOCAL_DEVICES = [x.name for x in devices]
    # return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

def split_convert(word_inp): 
    return [ord(i) for i in word_inp]

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


def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost

def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost

def f1_loss_3(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

class Model_Connected():
    def __init__(self, hiddensize, seed):
        self.hiddensize = hiddensize
        self.seed = seed

        self.model = Sequential()
        self.model.add(Dense(HIDDENSIZE, input_dim = 80, activation='relu', kernel_initializer= initializers.lecun_normal(seed=SEED)))
        self.model.add(Dense(107, input_dim = HIDDENSIZE, activation='relu', kernel_initializer= initializers.lecun_normal(seed=SEED)))
        self.model.add(Dropout(rate = 0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1, input_dim = 107, activation='sigmoid'))

        self.opt = keras.optimizers.Adam()  # alternative: opt = SGD(lr=0.01, momentum=0.9)

    def compile(self):
        self.model.compile(optimizer=self.opt, loss="binary_crossentropy", metrics=['accuracy', get_f1])
        # self.model.compile(optimizer=self.opt, loss='mean_squared_error', metrics=['accuracy', get_f1])
        return self.model

    def train(self, X):
        pass

if __name__ == "__main__":
    # defining global variables
    NEPOCHS = 120
    BATCHSIZE = 80
    VALIDATIONSPLIT = 0.2
    HIDDENSIZE = 80
    SEED = 42

    """#### Load data & data inspection"""
    dat_train = pd.read_csv("./data/train.csv")
    dat_test = pd.read_csv("./data/test.csv")
    # check class balance on activation
    dat_train['Active'].value_counts()

    """#### Pre-process data"""
    train_seqs = [split_convert(i) for i in dat_train.iloc[:,0]]
    train_labels = [i for i in dat_train.iloc[:,1]]
    test_seqs = [split_convert(i) for i in dat_test.iloc[:,0]]

    kfold_splits = 10
    folds = list(StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=SEED).split(train_seqs, train_labels))

    """ binary/one-hot encoding"""
    onehot_encoder = OneHotEncoder(sparse=False)
    train_seqs_onehot = onehot_encoder.fit_transform(train_seqs)
    test_seqs_onehot = onehot_encoder.transform(test_seqs)

    """ Create Model"""
    model_i = Model_Connected(HIDDENSIZE, SEED)
    model = model_i.compile()
    model.summary()

    best_fold = -1
    best_score = 0
    best_model = None

    """ Train model and pick best fold, TODO:Add hyperparameters that change for each fold """
    for index, (train_indices, val_indices) in enumerate(folds):
        print("Training on fold " + str(index+1) + "/10...")
        # Generate batches from indices
        xtrain, xval = train_seqs_onehot[train_indices], train_seqs_onehot[val_indices]
        #ytrain, yval = train_labels_onehot[train_indices], train_labels_onehot[val_indices]
        ytrain = np.array(train_labels)[train_indices.astype(int)]
        yval = np.array(train_labels)[val_indices.astype(int)]

        # determine class imbalance
        class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
        class_weight_dict = dict(enumerate(class_weights))

        # train
        model = model_i.compile()  # better way to just wipe model?
        model.fit(xtrain, ytrain, validation_data = (xval, yval), epochs = NEPOCHS, batch_size=BATCHSIZE, verbose = 0 , class_weight = class_weight_dict)  # starts training

        # predictions
        y_pred = model.predict_classes(xval, batch_size=BATCHSIZE, verbose=1)
        y_train = model.predict_classes(xtrain, batch_size=BATCHSIZE, verbose=1)
        y_pred_bool = y_pred.astype(int)
        tmp_score = metrics.f1_score(yval, y_pred)
        score_train = metrics.f1_score(ytrain, y_train)
        print("F1 score for this fold is : ", tmp_score, score_train)
        if (tmp_score > best_score):
            best_fold = index
            best_model = model

    """ train model on entire data set"""
    # class weight for the train set
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    class_weight_dict = dict(enumerate(class_weights))
    model.fit(train_seqs_onehot,train_labels, validation_split=0, epochs = NEPOCHS, batch_size=BATCHSIZE, verbose = 0, class_weight = class_weight_dict)  # starts training

    # Training Error
    y_pred = model.predict_classes(train_seqs_onehot, batch_size=BATCHSIZE, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(train_labels, y_pred))

    """#### Prediction on test data"""
    y_pred = model.predict_classes(test_seqs_onehot, batch_size=BATCHSIZE,verbose = 1)
    print(np.sum(y_pred))
    res = pd.dataFrame(y_pred)

    """ Save results"""
    import time, datetime
    timestamp = int(datetime.datetime.now())
    res.to_csv(f"./results/results_{timestamp}.csv", index=False, header=False)

