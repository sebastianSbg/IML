# -*- coding: utf-8 -*-
"""main_project4_7.ipynb

Created by Sebastian Bommer 2021_05_19

"""
# import cPickle
import time
import datetime
import math
import logging
import pandas as pd
import _pickle as pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0,1.2,3
import keras
import tensorflow as tf
import cv2

import random
from sklearn.model_selection import train_test_split
from keras.models import load_model

# # from google.colab import files
# import io
# import time

# import pdb

# import matplotlib.pyplot as plt

# import keras.backend as K
# from keras import Sequential

# from keras.layers import Dense, Dropout, Activation, SimpleRNN, LSTM, Conv1D, MaxPooling1D, AveragePooling1D, Embedding
# from keras.layers import BatchNormalization
# from keras.layers import AlphaDropout
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Lambda, Input, Layer

# from keras.regularizers import l2

# from keras.models import Model

# from keras import regularizers
# from keras.optimizers import SGD
# from keras.callbacks import Callback,ModelCheckpoint, EarlyStopping
# from keras import initializers

from matplotlib import pyplot as plt


"""FUNCTION DEFINITIONS"""


def similar_dish(prediction):
    out = [0] * len(prediction)
    for idx, vec in enumerate(prediction):
        anchor, positive, negative = vec[:SIZE_BOTTLENECK], vec[SIZE_BOTTLENECK:2 * SIZE_BOTTLENECK], vec[
                                                                                                      2 * SIZE_BOTTLENECK:]
        positive_dist = np.mean(np.square(anchor - positive))
        negative_dist = np.mean(np.square(anchor - negative))
        if positive_dist < negative_dist:
            out[idx] = 1
        else:
            out[idx] = 0
    return out


# # less memory intense way of creating predictions
# def predict_vec(fold, batch_size=1000, toSize=(450, 300), flatten=True):
#     if flatten:
#         x_anchors = np.zeros((batch_size, toSize[0] * toSize[1] * 3))
#         x_positives = np.zeros((batch_size, toSize[0] * toSize[1] * 3))
#         x_negatives = np.zeros((batch_size, toSize[0] * toSize[1] * 3))
#     else:
#         x_anchors = np.zeros([batch_size, toSize[1], toSize[0], 3])
#         x_positives = np.zeros([batch_size, toSize[1], toSize[0], 3])
#         x_negatives = np.zeros([batch_size, toSize[1], toSize[0], 3])
#     predictions = []
#     c = 0
#     totlen = 0
#     # get random samples from train triplets
#     for i in range(0, len(fold)):
#         if i % 500 == 0:
#             print(f'The progress is {round(i / len(fold) * 100, 1)} %')
#         # preprocessing
#         # We need to find an anchor, a positive example and a negative example
#         triplet = fold[i]
#         # reshaping, normalizing and flattening images --> this will take more time than pre-processed data, but will eliminate RAM issues
#         x_anchor = cv2.resize(food_images[triplet[0]], toSize) / 255.0
#         x_positive = cv2.resize(food_images[triplet[1]], toSize) / 255.0
#         x_negative = cv2.resize(food_images[triplet[2]], toSize) / 255.0
#         if flatten:
#             x_anchor = np.reshape(x_anchor, np.prod(x_anchor.shape))
#             x_positive = np.reshape(x_positive, np.prod(x_positive.shape))
#             x_negative = np.reshape(x_negative, np.prod(x_negative.shape))
#         x_anchors[c] = x_anchor
#         x_positives[c] = x_positive
#         x_negatives[c] = x_negative
#         if c >= (batch_size - 1) or i == (len(fold) - 1):
#             if i == (len(fold) - 1):
#                 print(f'End with c {c}')
#             print(f'i is {i}, Predicting...')
#             # print(x_anchors[-1])
#             X = [x_anchors[0:c], x_positives[0:c], x_negatives[0:c]]
#             print(np.array(X).shape)
#             vec = net.predict(X)
#             predictions.append(similar_dish(vec))
#             # predictions.append(similar_dish(vec))
#             print('End of predictions...')
#             totlen = totlen + len(vec)
#             # resetting c
#             c = -1
#             # break
#         c = c + 1
#     pred_flat = []
#     for li in predictions:
#         for elem in li:
#             pred_flat.append(elem)

    # print(f' The length of the vector is: {totlen}')
    # return pred_flat



# def run_through(fold, toSize=(450, 300), flatten=True):
#     batch_size = len(fold)
#     if flatten:
#         x_anchors = np.zeros((batch_size, toSize[0] * toSize[1] * 3))
#         x_positives = np.zeros((batch_size, toSize[0] * toSize[1] * 3))
#         x_negatives = np.zeros((batch_size, toSize[0] * toSize[1] * 3))
#     else:
#         x_anchors = np.zeros([batch_size, toSize[1], toSize[0], 3])
#         x_positives = np.zeros([batch_size, toSize[1], toSize[0], 3])
#         x_negatives = np.zeros([batch_size, toSize[1], toSize[0], 3])

#     # get random samples from

# for i in range(0, len(fold)):
#     # preprocessing
#     # We need to find an anchor, a positive example and a negative example
#     triplet = fold[i]
#     # reshaping, normalizing and flattening images --> this will take more time than pre-processed data, but will eliminate RAM issues
#     x_anchor = cv2.resize(food_images[triplet[0]], toSize) / 255.0
#     x_positive = cv2.resize(food_images[triplet[1]], toSize) / 255.0
#     x_negative = cv2.resize(food_images[triplet[2]], toSize) / 255.0
#     if flatten:
#         x_anchor = np.reshape(x_anchor, np.prod(x_anchor.shape))
#         x_positive = np.reshape(x_positive, np.prod(x_positive.shape))
#         x_negative = np.reshape(x_negative, np.prod(x_negative.shape))

    # x_anchors[i] = x_anchor
    # x_positives[i] = x_positive
    # x_negatives[i] = x_negative
    # # yield [x_anchors, x_positives, x_negatives]

# return [x_anchors, x_positives, x_negatives]

# def plot_triplets(examples, imgSize):
#     plt.figure(figsize=(6, 2))
#     for i in range(3):
#         plt.subplot(1, 3, 1 + i)
#         plt.imshow(np.reshape(examples[i], (imgSize[1], imgSize[0], 3)))
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()

def create_batch(fold, img_dict, batch_size=16, toSize=(100, 100), flatten=False):
    if flatten:
        # toSize = img_dict[0][0].shape
        x_anchors = np.zeros((batch_size, toSize[0] * toSize[1] * 3))
        x_positives = np.zeros((batch_size, toSize[0] * toSize[1] * 3))
        x_negatives = np.zeros((batch_size, toSize[0] * toSize[1] * 3))
    else:
        x_anchors = np.zeros([batch_size, toSize[1], toSize[0], 3])
        x_positives = np.zeros([batch_size, toSize[1], toSize[0], 3])
        x_negatives = np.zeros([batch_size, toSize[1], toSize[0], 3])
    # get random samples from train triplets
    for i in range(0, batch_size):
        # preprocessing
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, fold.shape[0] - 1)
        triplet = fold[random_index]
        # logger.debug(f"Triplet: {triplet}")
        # reshaping, normalizing and flattening images --> this will take more time than pre-processed data, but will eliminate RAM issues
        try:
            # x_anchor = cv2.resize(img_dict[int(triplet[0])], toSize) / 255.0
            x_anchor = img_dict[int(triplet[0])] / 255.0
            # x_positive = cv2.resize(img_dict[int(triplet[1])], toSize) / 255.0
            x_positive = img_dict[int(triplet[1])] / 255.0
            # x_negative = cv2.resize(img_dict[int(triplet[2])], toSize) / 255.0
            x_negative = img_dict[int(triplet[2])]/ 255.0
        except KeyError:
            logger.exception(f"One of the keys was not found in the image dictionary {int(triplet[0])}, {int(triplet[1])}, {int(triplet[2])} not found")
        except:
            logger.exception("Error getting and resizing images")
        if flatten:
            x_anchor = np.reshape(x_anchor, np.prod(x_anchor.shape))
            x_positive = np.reshape(x_positive, np.prod(x_positive.shape))
            x_negative = np.reshape(x_negative, np.prod(x_negative.shape))
        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
        # logger.debug(f"X_anchor shape: {x_anchors.shape}")
    return [x_anchors, x_positives, x_negatives]  # TODO: remove np.array conversion?

def data_generator(fold, img_dict, batch_size=4, toSize=(100,100), flatten=True):
    while True:
        X = create_batch(fold, img_dict, batch_size, toSize, flatten)
        y = np.zeros((batch_size, 3 * SIZE_BOTTLENECK))
        # logger.debug(f"Generator called with shape {np.array(X).shape}")
        yield X, y

def get_pp_images(img_size = [100,100], maxImages=10000, img_path=None,  force_recompute=False, savePP = True):  # TODO: Improve reading images with LMDB and HDF5
    """ Function applies downsampling and preprocessing to original images. The processed data can be saved in a pickel item and reloaded for future runs."""

    out_dic = {}

    if img_path == None:
        pp_path = os.path.join('data', 'img_dic.pickle')  # preprocessed pickle object path
    else:
        pp_path = img_path

    if os.path.exists(pp_path) and not force_recompute:
        logger.info("Loading preprocessed data ... ")
        try:
            with open(pp_path, "rb") as input_file:
                pickle_in = pickle.load(input_file)
                return pickle_in
            logger.info("The image array was successfully loaded")
        except:
            logger.error("The image array failed to load")
    else:
        logger.info("Computing preprocessed data .. ")
        idx = 0
        img_name = lambda idx: os.path.join('data', 'img', f'{idx:05d}.jpg')
        while True and idx < maxImages:
            if idx%500==0:logger.debug(f"Image name: {img_name(idx)}")
            try:
                if not os.path.exists(img_name(idx)):
                    raise FileExistsError

                img = cv2.imread(img_name(idx))  # resize image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # resize image
                img = cv2.resize(img, img_size)  # appending image
                out_dic[idx] = img

                idx += 1

            except:
                logger.exception(f"File does not exist, import was terminated at index {idx}")
                break

        if savePP:
            try:
                with open(pp_path, "wb") as output_file:
                    pickle.dump(out_dic, output_file)
                logger.info("The image array was successfully saved!")
            except:
                logger.exception("The image array failed to save!")

    return out_dic

def myLogger(logLevel, fileLogLevel = logging.WARNING):
    path = os.path.dirname(__file__)
    logger = logging.getLogger(__name__)
    logger.setLevel(logLevel)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    file_handler_e = logging.FileHandler(f'log.log')
    file_handler_e.setLevel(fileLogLevel)
    file_handler_e.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler_e)
    logger.addHandler(stream_handler)
    return logger

def triplet_loss(y_true, y_pred):
   anchor, positive, negative = y_pred[:, :SIZE_BOTTLENECK], y_pred[:, SIZE_BOTTLENECK:2 * SIZE_BOTTLENECK], y_pred[:, 2 * SIZE_BOTTLENECK:]
   positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
   negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
   triplet_loss = tf.maximum(positive_dist - negative_dist + ALPHA, 0.)
   loss = tf.reduce_mean(triplet_loss)
   return loss

class siameseNetwork():
    def __init__(self, imgSize):
        self.net = None
        self.imgSize = imgSize
        self.get_network()
        self.trainingData = None
        # self.trained_model = None
    def embedding_model(self):
        embedding_model = keras.Sequential()
        embedding_model.add(keras.Input(shape=(IMG_SIZE_IN[1], IMG_SIZE_IN[0], 3)))
        embedding_model.add(tf.keras.layers.Conv2D(32, 5, strides=2, activation="relu"))
        embedding_model.add(tf.keras.layers.MaxPooling2D(2))
        embedding_model.add(tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu"))
        embedding_model.add(tf.keras.layers.MaxPooling2D(2))
        # embedding_model.add(layers.GlobalMaxPooling2D())
        embedding_model.add(tf.keras.layers.Flatten())
        embedding_model.add(tf.keras.layers.Dense(SIZE_BOTTLENECK, activation='sigmoid'))
        return embedding_model
    def get_network(self):
        input_anchor = tf.keras.layers.Input(shape=(self.imgSize[1], self.imgSize[0], 3))
        input_positive = tf.keras.layers.Input(shape=(self.imgSize[1], self.imgSize[0], 3))
        input_negative = tf.keras.layers.Input(shape=(self.imgSize[1], self.imgSize[0], 3))

        embedding_model = self.embedding_model()
        embedding_anchor = embedding_model(input_anchor)
        embedding_positive = embedding_model(input_positive)
        embedding_negative = embedding_model(input_negative)

        output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)
        self.net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)

    def get_model_summary(self):
        self.net.summary()
        tf.keras.utils.plot_model(self.net, show_shapes=True, show_layer_names=True)

    def train(self, data_in, img_dict, epochs = 20, batch_size = 8, saveModel = True):
        # keras.losses.custom_loss = self.triplet_loss
        # if loss:
        #     loss = self.triplet_loss()
        keras.losses.custom_loss = triplet_loss

        self.net.compile(loss = triplet_loss, optimizer = 'adamax')  #TODO: Change optimizer to 'adamax' + define custom loss to be triplet loss
        try:
            self.net.fit(data_generator(data_in, img_dict, batch_size = batch_size, flatten=False), epochs=epochs, steps_per_epoch = math.floor(data_in.shape[0] / batch_size))
            self.timeCompleted = int(time.time())
            readableTime = str(datetime.datetime.fromtimestamp(self.timeCompleted).strftime('%c'))
            logger.info(f'Finished training at {readableTime}')
        except:
            file_path = os.path.join('models', f"interrupted_model_{self.timeCompleted}.h5")
            self.net.save_weights(file_path)
            logger.exception("Siamese Network failed to train!")

        if saveModel:
            self.saveModel()

    def saveModel(self):
        try:
            file_path = os.path.join('models', f"model_{self.timeCompleted}.h5")
            # file_path_weights = os.path.join('models', f"model_weights_{self.timeCompleted}.h5")
            logger.debug(f"File name: {file_path}")
            self.net.save(file_path)
            # self.net.save_weights(file_path_weights)
            # tf.keras.models.save_model(self.net, file_path)
            logger.info(f"Model was saved successfully.")
        except FileNotFoundError:
            logger.error(f"Model failed to save to file, because file was not found")
        except:
            logger.exception(f"Failed to save model!")

    def loadModel(self):
        pass

    def inference(self, data, imgDict, savePredictions = True, path = None):
        # generator to avoid having to load everything into memory
        inferenceGenerator = ([np.array([imgDict[triplet[0]]/255.0]), np.array([imgDict[triplet[1]]/255.0]), np.array([imgDict[triplet[2]]/255.0])] for triplet in data)  #TODO: Numpy array conversions to be done in preprocessing steps

        # logger.debug(f"{next(inferenceGenerator)}")

        X_b = create_batch(data[:4], imgDict)
        logger.debug(f"Shape of create_batch: {np.array(X_b).shape}")
        logger.debug(f"Type of create_patch {type(X_b)}")

        try:
            logger.info(f"Inference started ...")
            # self.net.predict(inferenceGenerator)
            X = [np.array([imgDict[0] / 255.0]), np.array([imgDict[15] / 255.0]), np.array([imgDict[1000] / 255.0])]
            logger.debug(f"Predict input shape: {np.array(X).shape}")
            logger.debug(f"Type of my input {type(X)}")
            logger.debug(f"Image shape: {(imgDict[0]/255.0).shape}")
            logger.debug(f"inference shape: {np.array(next(inferenceGenerator)).shape}")
            self.predictions = self.net.predict_generator(inferenceGenerator)
            # self.net.predict(X)
        except:
            logger.exception(f"Inference failed!")

        if savePredictions:
            self.savePredictions(path)

    def savePredictions(self, path = None):
        if path == None:
            file_path = os.path.join('results', f"results_{int(time.time())}")
        else:
            file_path = path

        try:
            sv_pr = similar_dish(self.predictions)
            np.savetxt(file_path, sv_pr, fmt='%i')
            logger.info("The predictions were successfully saved ... ")
        except:
            logger.exception("The predictions couldn't be saved!")


    def loadLatestModel(self):
        model_path = os.path.join('models', max(os.listdir('./models')))
        logger.info(f"Model Path: {model_path}")
        try:
            self.net = load_model(model_path, custom_objects={"triplet_loss": triplet_loss})
            self.net.summary()
            logger.info(f"Successfully loaded in model {model_path}")
        except FileNotFoundError:

            logger.error(f"Model was not found: {model_path} does not exist")
        except:
            logger.exception(f"Latest Model could not be loaded!")
        return self.net


if __name__ == "__main__":

    """Hyper parameters"""
    IMG_SIZE_IN = (100, 100) # desired image size for input to layer in format (w,h)
    SIZE_BOTTLENECK = 50
    ALPHA = 1.0 # margin for triplet loss
    BATCH_SIZE = 64
    EPOCHS = 10
    LOG_LEVEL = logging.DEBUG
    INPUT_ON = True # if False the user can intervene using inputs
    TEST_SIZE = 0.2

    """Importing and preprocessing"""

    logger = myLogger(LOG_LEVEL)

    img_dic = get_pp_images(maxImages=10000, force_recompute=False)

    # """## Importing training and testing triplets"""

    try:
        X_df = pd.read_csv(os.path.join('data', 'train_triplets.txt'), header = None, delim_whitespace=True)
        n_data = X_df.shape[0]
        X_predict_df = pd.read_csv(os.path.join('data', 'test_triplets.txt'), header = None, delim_whitespace=True)
        logger.info(f"Successfully imported .txt file")
    except FileNotFoundError:
        logger.error(".txt file was not found!")
    except:
        logger.exception("Couldn't read the .txt file")

    # train_triplets = np.array(X_train_df)
    # test_triplets = np.array(X_test_df)
    # n_train = np.shape(X_train_df)[0]
    # t_test = np.shape(X_test_df)[0]

    X = np.array(X_df)  # TODO: create a validation set
    X_pred = np.array(X_predict_df)
    y = np.ones(X.shape[0])

    X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=TEST_SIZE, random_state=1)

    if INPUT_ON and input("Do you want to perform inference only?: YES for INFERENCE Y/N: ").lower() in ['y','yes','1']:
        # TODO: import model and create inference
        logger.info(f"INFERENCE ONLY ... ")
        net = siameseNetwork(IMG_SIZE_IN)
        net.loadLatestModel()

        net.inference(X_pred, img_dic, savePredictions=True)
    else:
        logger.info(f"TRAINING and INFERENCE ... ")

        net = siameseNetwork(IMG_SIZE_IN)
        net.get_model_summary()
        net.train(X_train, img_dic, epochs=EPOCHS)

        # file_path = os.path.join('results', f"results_{int(time.time())}")
        # predicted_embeddings = net.inference(X_pred, img_dic)
        # sv_pr = similar_dish(predicted_embeddings)
        # np.savetxt(file_path, sv_pr, fmt='%i', )

        # dg = data_generator(X_train, img_dic)
        # logger.debug(len(next(dg)))
        logger.info(f"Training and Inference ended.")











































# """# Test predictions & save results"""

# # optionally loading a model
# # from tensorflow.keras.models import load_model

# # inp = input('Select model to run: ')

# # net_saved = load_model(inp, custom_objects={'triplet_loss': triplet_loss})

# # function for transforming predicted embeddings to required prediction 0 or 1
# # anchor: dish A, positive: dish B, negative: dish C
# # if distance A to B < A to C then A is more similar to B than to C -> return 1

# #--> debug predict_vec (the output sizes are off by 1)
# # pr_val = predict_vec(validate, batch_size = 501, toSize = IMG_SIZE_IN, flatten=False)

# # # There is still a discrepancy
# # print(len(outVec))
# # print(len(validate))

# # # percentage accuracy
# # perc_accuracy = round(np.sum(pr_val)/len(pr_val)*100,1)
# # print(f'The accurate predictions from the validation set is {perc_accuracy} %')

# #deprecated

# # X = run_through(validate, toSize = IMG_SIZE_IN, flatten=False)

# # predicted_embeddings = net.predict(X)
# # predicted_embeddings.shape

# # making predictions for the test set in the first fold
# # sv_f1 = [similar_dish(emb) for emb in predicted_embeddings]
# # perc_accurate = round(np.sum(sv_f1)/len(sv_f1)*100,1)
# # print(f'The percentage of accurate predictions for the first fold test set is: {perc_accurate} %')

# # pr_test = predict_vec(D_test, batch_size = 1000, toSize = IMG_SIZE_IN, flatten=False)

# # # percentage accuracy
# # positive_given = round(np.sum(pr_test)/len(pr_test)*100,1)
# # print(f'Predicted positive images given is {positive_given} %')

# # deprecated
# D_test = np.array(D_test)
# X = run_through(D_test, toSize = IMG_SIZE_IN, flatten=False)

# predicted_embeddings = net.predict(X)
# predicted_embeddings.shape

# # # making predictions for the test set in the first fold
# sv_pr = similar_dish(predicted_embeddings)
# perc_accurate = round(np.sum(sv_pr)/len(sv_pr)*100,1)
# print(f'The percentage of accurate predictions for the test set is: {perc_accurate} %')
# print(np.array(sv_pr).shape)

# description = input('Add an optional description to the result file: ')
# filepath = f'{workDir}/results/{round(now.timestamp())}_sb_result_{description}.txt'
# np.savetxt(filepath, sv_pr, fmt='%i', )