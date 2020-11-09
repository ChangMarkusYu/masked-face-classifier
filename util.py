'''
Utilty functions for pca-svm pipeline and CNN
'''

import os
import glob
import math
import cv2
import time
import pickle
import numpy as np
from joblib import Parallel, delayed, dump
from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce
from scipy import stats

J = 0  # Just load folder 00000 for now

'''
Read image as gray scale image (1024, 1024). Downsample it to (256,256)
by averaging the pixel values in each 4 by 4 pixel block. Finally,
flatten the 2d matrix into a 1d vector.

Grayscale and downsamping is a must, otherwise the memory cost will be too huge
and blow up my computer
'''


def preprocess_image(im_file):
    img = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    img = block_reduce(img, block_size=(8, 8), func=np.mean).flatten()
    return stats.zscore(img)


'''
Load the images of a particular class label into memory
The shape of the feature matrix X follows the normal convention: (n_samples, n_features)
'''


def load(class_label):
    X = []  # input features
    Y = []  # class labels
    input_folder = "CMFD" if class_label == 1 else "IMFD"
    # load correctly masked faces
    print(f"Loading class: {class_label}")
    for j in range(J + 1):
        folder_num = j * 10
        folder_name = format(folder_num, '05d')
        print("Now Loading: " + folder_name)
        path = os.path.join(f'./{input_folder}/', folder_name, '*.jpg')
        files = glob.glob(path)
        X.extend(Parallel(n_jobs=-2)(delayed(preprocess_image)(im_file)
                                     for im_file in files))
        X = np.asarray(X)
        Y = np.asarray([class_label] * len(files))
        return X, Y


'''
Load image data and shuffle. Use 80/20 train-test split
'''


def load_data():
    X1, Y1 = load(1)
    X2, Y2 = load(0)
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=True)
    return X_train, X_test, y_train, y_test


'''
Save the model as a file to ./model
model_type: saving the pca-svm pipeline or the cnn. Can only be one of 'pca' or 'cnn'

To load the saved model, import load from joblib and then use: clf = load('filepath.joblib')
'''


def model_checkpoint(model, model_type='pca'):
    if model_type != 'pca' and model_type != 'cnn':
        raise ValueError("model_type must be either 'pca' or 'cnn'")

    timestr = time.strftime("%Y%m%d_%H%M%S")
    filename = f"model_{model_type}_{timestr}.joblib"
    filepath = os.path.join('./model', filename)
    dump(model, filepath)
