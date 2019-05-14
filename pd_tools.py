# -*- coding: utf-8 -*-
# Author: Zihao Wang

from skimage import io
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import math
import numpy as np
from skimage import exposure, img_as_float
from sklearn.model_selection import GridSearchCV


# Avatar Image Save Path
profile_image_path = '../img/'


def inspect_dataset(df_data):

    print('Dataset Basic Information：')
    print(df_data.info())
    print('Dataset Has {} Rows，{} Columns'.format(df_data.shape[0], df_data.shape[1]))
    print('Data Preview:')
    print(df_data.head())


def check_profile_image(img_link):
    """
        Check Whether Avatar Image Link is Valid or Not
        If Valid, Download to the Local Computer, and Return Save Path
    """
    save_image_path = ''
    # Valid Image File Extension:
    valid_img_ext_lst = ['.jpeg', '.png', '.jpg']

    try:
        img_data = io.imread(img_link)
        image_name = img_link.rsplit('/')[-1]
        if any(valid_img_ext in image_name.lower() for valid_img_ext in valid_img_ext_lst):
            # Make Sure the Image File Contains a Valid Extension
            save_image_path = os.path.join(profile_image_path, image_name)
            io.imsave(save_image_path, img_data)
    except:
        print('Avatar Image Link {} Invalid'.format(img_link))

    return save_image_path


def clean_text(text):

    # just in case
    text = text.lower()

    # Remove Special Characters
    text = re.sub('\s\W', ' ', text)
    text = re.sub('\W\s', ' ', text)
    text = re.sub('\s+', ' ', text)

    return text


def split_train_test(df_data, size=0.8):

    # To Ensure that the Data in Each Class Can be the Same Proportion in the Training Set and the Test Set, So Each Class Needs to be Processed in Turn
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    labels = [0, 1]
    for label in labels:
        # Find out Data Record of Gender
        text_df_w_label = df_data[df_data['label'] == label]
        # Reset Index to Ensure the Data Record of Each Class Starts From Index 0, Convenient for Future Split
        text_df_w_label = text_df_w_label.reset_index()

        # Default 80% Training Set and 20% Test Set Split

        # The Number of Rows of That Class Data 
        n_lines = text_df_w_label.shape[0]
        split_line_no = math.floor(n_lines * size)
        text_df_w_label_train = text_df_w_label.iloc[:split_line_no, :]
        text_df_w_label_test = text_df_w_label.iloc[split_line_no:, :]

        # Put Them to Overall Training Dataset and Test Dataset
        df_train = df_train.append(text_df_w_label_train)
        df_test = df_test.append(text_df_w_label_test)

    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    return df_train, df_test


def get_word_list_from_data(text_s):
    """
        Put the Words in the Dataset Into a List
    """
    word_list = []
    for _, text in text_s.iteritems():
        word_list += text.split(' ')
    return word_list


def proc_text(text):
    """
        Tokenization+Remove Stopwords
    """
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(filtered_words)

def extract_tf_idf(text_s, text_collection, common_words_freqs):
    """
        Extract TF-IDF Feature
    """


    n_sample = text_s.shape[0]
    n_feat = len(common_words_freqs)

    common_words = [word for word, _ in common_words_freqs]

    # Initialization
    X = np.zeros([n_sample, n_feat])

    print('Extract TF-IDF Feature...')
    for i, text in text_s.iteritems():
        feat_vec = []
        for word in common_words:
            if word in text:
                # If in High-Frequency Words，Calculate TF-IDF Value
                tf_idf_val = text_collection.tf_idf(word, text)
            else:
                tf_idf_val = 0

            feat_vec.append(tf_idf_val)

        # Assignment
        X[i, :] = np.array(feat_vec)

    return X


def hex_to_rgb(value):
    """
        Convert Hex Clor to RGB Vaule
    """
    rgb_list = list(int(value[i:i + 2], 16) for i in range(0, 6, 2))
    return rgb_list


def extract_rgb_feat(hex_color_s):
    """
        Extract RGB Value from Hex Color as Feature
    """
    n_sample = hex_color_s.shape[0]
    n_feat = 3

    # Initialization
    X = np.zeros([n_sample, n_feat])

    print('Extract RGB Feature...')
    for i, hex_val in hex_color_s.iteritems():
        feat_vec = hex_to_rgb(hex_val)

        # Assignment
        X[i, :] = np.array(feat_vec)

    return X


def extract_rgb_hist_feat(img_path_s):
    """
        Extract RGB Histogram Feature from Image
    """
    n_sample = img_path_s.shape[0]
    n_bins = 100    # The Number of Bins in Each Channel
    n_feat = n_bins * 3

    # Initialization
    X = np.zeros([n_sample, n_feat])

    print('Extract RGB Histogram Feature...')
    for i, img_path in img_path_s.iteritems():
        # Loading Image
        img_data = io.imread(img_path)
        img_data = img_as_float(img_data)

        if img_data.ndim == 3:
            # 3 Channels
            hist_r, _ = exposure.histogram(img_data[:, :, 0], nbins=n_bins)
            hist_g, _ = exposure.histogram(img_data[:, :, 1], nbins=n_bins)
            hist_b, _ = exposure.histogram(img_data[:, :, 2], nbins=n_bins)
        else:
            # 2 Channels
            hist, _ = exposure.histogram(img_data, nbins=n_bins)
            hist_r = hist.copy()
            hist_g = hist.copy()
            hist_b = hist.copy()

        feat_vec = np.concatenate((hist_r, hist_b, hist_g))

        # Assignment
        X[i, :] = np.array(feat_vec)

    return X

def get_best_model(model, X_train, y_train, params, cv=5):
    """
        Grid Search to Obtain Optimal Model
        Default 5-Fold-Cross-Validation
    """
    clf = GridSearchCV(model, params, cv=cv, verbose=3)
    clf.fit(X_train, y_train)
    return clf.best_estimator_
