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
* implement progress bar
"""
import logging
import os
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
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

# %tensorflow_version 2.x
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
# # import keras.backend.tensorflow_backend as tfback
# from tensorflow.python.client import device_lib
#

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
        self.model.compile(optimizer=self.opt, loss="binary_crossentropy", metrics=['accuracy', get_f1])
        # self.model.compile(optimizer=self.opt, loss='mean_squared_error', metrics=['accuracy', get_f1])
        return self.model

    def train(self, X):
        pass

if __name__ == "__main__":
    # defining global variables
    NEPOCHS = 200
    BATCHSIZE = 50
    VALIDATIONSPLIT = 0.2
    HIDDENSIZE = 200
    SEED = 29
    KFOLDSPLITS = 2
    VERBOSE = True
    LOGLEVEL = logging.DEBUG

    # define logger
    logger = myLogger(LOGLEVEL)

    """#### Load data & data inspection"""
    dat_train = pd.read_csv("./data/train.csv")
    dat_test = pd.read_csv("./data/test.csv")
    logger.info(f"Pecenttage of active values: {np.round(100*dat_train['Active'].value_counts()/dat_train.shape[0],4)}")  # check class balance on activation

    """#### Pre-process data"""
    train_seqs = [split_convert(i) for i in dat_train.iloc[:,0]]  # convestion not necessary
    train_labels = [i for i in dat_train.iloc[:,1]]
    test_seqs = [split_convert(i) for i in dat_test.iloc[:,0]]

    folds = list(StratifiedKFold(n_splits=KFOLDSPLITS, shuffle=True, random_state=SEED).split(train_seqs, train_labels))

    """ binary/one-hot encoding"""  #TODO: Check if it would be better to eliminate one hot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    train_seqs_onehot = onehot_encoder.fit_transform(train_seqs)
    test_seqs_onehot = onehot_encoder.transform(test_seqs)

    """ Create Model"""
    model_inst = Model_Connected(HIDDENSIZE, SEED)
    model = model_inst.compile()
    model.summary()

    best_fold = -1
    best_score = 0
    best_model = None

    """Train Model"""
    xtrain, xval = train_seqs_onehot, train_seqs_onehot
    #ytrain, yval = train_labels_onehot[train_indices], train_labels_onehot[val_indices]
    ytrain = np.array(train_labels)
    yval = np.array(train_labels)
    # determine class imbalance
    logger.debug("Class imbalance start...")
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    class_weight_dict = dict(enumerate(class_weights))
    logger.debug("Class imbalance end")
    # train
    logger.debug("Model compile start...")
    model = model_inst.compile()
    logger.debug("Model train start ...")
    model.fit(xtrain, ytrain, validation_data = (xval, yval), epochs = NEPOCHS, batch_size=BATCHSIZE, verbose = VERBOSE , class_weight = class_weight_dict)  # starts training
    # predictions
    logger.debug("Model predictions start...")
    y_pred = model.predict_classes(xval, batch_size=BATCHSIZE, verbose=1)
    y_train = model.predict_classes(xtrain, batch_size=BATCHSIZE, verbose=1)
    y_pred_bool = y_pred.astype(int)
    tmp_score = metrics.f1_score(yval, y_pred)
    score_train = metrics.f1_score(ytrain, y_train)
    logger.info("F1 score for this fold is : ", tmp_score, score_train)
    best_model = model

    """#### Prediction on test data"""
    y_pred = best_model.predict_classes(test_seqs_onehot, batch_size=BATCHSIZE,verbose = VERBOSE)
    logger.info(np.sum(y_pred))
    res = pd.DataFrame(y_pred)

    """ Save results"""
    import time
    timestamp = int(time.time())
    res.to_csv(f"./results/results_{timestamp}.csv", index=False, header=False)

