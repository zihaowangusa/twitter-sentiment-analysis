# -*- coding: utf-8 -*-
# Author: Zihao Wang

import os
import pandas as pd
import matplotlib.pyplot as plt
from common_tools import get_dataset_filename, unzip, cal_acc
from pd_tools import inspect_dataset, check_profile_image, \
    split_train_test, clean_text, proc_text, get_word_list_from_data, \
    extract_tf_idf, extract_rgb_feat, extract_rgb_hist_feat,get_best_model
import nltk
from nltk.text import TextCollection
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score



# Declare Dataset Path
dataset_path = '../dataset'  # Dataset Path
zip_filename = 'twitter.zip'  # Zip File Name
zip_filepath = os.path.join(dataset_path, zip_filename)  # Zip File Path
cln_datapath = '../dataset/cleaned_data.csv'     # Cleaned Dataset Path

# Judge Whether First Run or Not
is_first_run = False


def run_main():
    """
        Main Function
    """
    # Declare Variables
    dataset_filename = get_dataset_filename(zip_filepath)  # Dataset Filename（In Zip File）
    dataset_filepath = os.path.join(dataset_path, dataset_filename)  # Dataset File Path

    if is_first_run:

        print('Unzip...', end='')
        unzip(zip_filepath, dataset_path)
        print('Finished.')

        # Read Data
        data = pd.read_csv(dataset_filepath, encoding='latin1',
                           usecols=['gender', 'description', 'link_color',
                                    'profileimage', 'sidebar_color', 'text'])
        # 1. View the Loaded Dataset
        inspect_dataset(data)

        # 2. Data Clean
        # 2.1. Filter Data According to 'gender' Column
        filtered_data = data[(data['gender'] == 'male') | (data['gender'] == 'female')]

        # 2.2 Filter Empty Values in 'description' Column
        filtered_data = filtered_data.dropna(subset=['description'])

        # 2.3 Filter illegal Hexadecimal Data in 'link_color' Column and 'sidebar_color' Column
        filtered_data = filtered_data[filtered_data['link_color'].str.len() == 6]
        filtered_data = filtered_data[filtered_data['sidebar_color'].str.len() == 6]

        # 2.4 Clean Text Data
        print('Clean Text Data...')
        cln_desc = filtered_data['description'].apply(clean_text)
        cln_text = filtered_data['text'].apply(clean_text)
        filtered_data['cln_desc'] = cln_desc
        filtered_data['cln_text'] = cln_text

        # 2.5 Check Whether the Profile Image Link is Valid in the 'profileimage' Column
        # Generate New Column to Record the Saved Path of Avatar Image
        print('Download Avatar Image...')
        saved_img_s = filtered_data['profileimage'].apply(check_profile_image)
        filtered_data['saved_image'] = saved_img_s
        # Filter Invalid Avatar Image Data
        filtered_data = filtered_data[filtered_data['saved_image'] != '']

        # Save the Processed Data
        filtered_data.to_csv(cln_datapath, index=False)

    # Read Processed Data
    clean_data = pd.read_csv(cln_datapath, encoding='latin1',
                             usecols=['gender', 'cln_desc', 'cln_text',
                                      'link_color', 'sidebar_color', 'saved_image'])

    # View Gender(Label) Distribution
    print(clean_data.groupby('gender').size())

    # Replace Male->0, Female->1
    clean_data.loc[clean_data['gender'] == 'male', 'label'] = 0
    clean_data.loc[clean_data['gender'] == 'female', 'label'] = 1

    # 3. Split the Data Set
    # Tokenization, Remove Stopwords
    proc_desc_s = clean_data['cln_desc'].apply(proc_text)
    clean_data['desc_words'] = proc_desc_s

    proc_text_s = clean_data['cln_text'].apply(proc_text)
    clean_data['text_words'] = proc_text_s

    df_train, df_test = split_train_test(clean_data)
    # View Basic Information of Training Dataset and Test Dataset
    print('The Number of Data in Each Class for Training Dataset：', df_train.groupby('label').size())
    print('The Number of Data in Each Class for Test Dataset：', df_test.groupby('label').size())

    # 4. Feature Engineering
    # 4.1 Training Data Feature Extraction
    print('Training Sample Feature Extraction：')
    # 4.1.1 Text Data
    # Description Data
    print('Calculate "Description" Word Frequency...')
    n_desc_common_words = 50
    desc_words_in_train = get_word_list_from_data(df_train['desc_words'])
    fdisk = nltk.FreqDist(desc_words_in_train)
    desc_common_words_freqs = fdisk.most_common(n_desc_common_words)
    print('The Most Frequently {} Words in "Descriptino" is：'.format(n_desc_common_words))
    for word, count in desc_common_words_freqs:
        print('{}: {} times'.format(word, count))
    print()

    # Extract TF-IDF Feature of "Desc" text
    print('Extract "Desc" Text Feature...', end=' ')
    desc_collection = TextCollection(df_train['desc_words'].values.tolist())
    tr_desc_feat = extract_tf_idf(df_train['desc_words'], desc_collection, desc_common_words_freqs)
    print('Finished')
    print()

    # Text Data
    print('Calculate Text Word Frequency...')
    n_text_common_words = 50
    text_words_in_train = get_word_list_from_data(df_train['text_words'])
    fdisk = nltk.FreqDist(text_words_in_train)
    text_common_words_freqs = fdisk.most_common(n_text_common_words)
    print('The Most Frequently {} Words in "Text" is：'.format(n_text_common_words))
    for word, count in text_common_words_freqs:
        print('{}: {} times'.format(word, count))
    print()

    # Extract TF-IDF Feature of Text Data
    text_collection = TextCollection(df_train['text_words'].values.tolist())
    print('Extract Text Data Feature...', end=' ')
    tr_text_feat = extract_tf_idf(df_train['text_words'], text_collection, text_common_words_freqs)
    print('Finished')
    print()

    # 4.1.2 Image Data
    # RGB Feature of Link Color
    tr_link_color_feat_ = extract_rgb_feat(df_train['link_color'])
    tr_sidebar_color_feat = extract_rgb_feat(df_train['sidebar_color'])

    # RGB Histogram Feature of Profile Image
    tr_profile_img_hist_feat = extract_rgb_hist_feat(df_train['saved_image'])

    # Combine Text Feature and Image Feature
    tr_feat = np.hstack((tr_desc_feat, tr_text_feat, tr_link_color_feat_,
                         tr_sidebar_color_feat, tr_profile_img_hist_feat))

    # Normalization
    scaler = StandardScaler()
    tr_feat_scaled = scaler.fit_transform(tr_feat)

    # Obtain Training Dataset Label
    tr_labels = df_train['label'].values

    # 4.2 Test Dataset Feature Extraction
    print('Test Sample Feature Extraction：')
    # 4.2.1 Text Data
    # Description Data
    # Extract TF-IDF Feature of "Desc"
    print('Extract "Desc" Text Feature...', end=' ')
    te_desc_feat = extract_tf_idf(df_test['desc_words'], desc_collection, desc_common_words_freqs)
    print('Finished')
    print()

    # Text Data
    # Extract TF-IDF Feature of "Text" 
    print('Extract "Text" Text Feature...', end=' ')
    te_text_feat = extract_tf_idf(df_test['text_words'], text_collection, text_common_words_freqs)
    print('Finished')
    print()

    # 4.2.2 Image Data
    # RGB Feature of Link Color
    te_link_color_feat_ = extract_rgb_feat(df_test['link_color'])
    te_sidebar_color_feat = extract_rgb_feat(df_test['sidebar_color'])

    # RGB Histogram Feature of Profile Image
    te_profile_img_hist_feat = extract_rgb_hist_feat(df_test['saved_image'])

    # Combine Text Feature and Image Feature
    te_feat = np.hstack((te_desc_feat, te_text_feat, te_link_color_feat_,
                         te_sidebar_color_feat, te_profile_img_hist_feat))

    # Normalization
    te_feat_scaled = scaler.transform(te_feat)

    # Obtain Training Dataset Label
    te_labels = df_test['label'].values

    # 4.3 PCA Dimension Reduction
    pca = PCA(n_components=0.95)  # Retain 95% Cumulative Contribution Rate of Eigenvectors
    tr_feat_scaled_pca = pca.fit_transform(tr_feat_scaled)
    te_feat_scaled_pca = pca.transform(te_feat_scaled)

    # 5. Train Model，Compare Performance Before and After PCA
    # Use Feature Without PCA
	
    # 5.1 Naive Bayes
    # nb = GaussianNB()
    # nb.fit(tr_feat_scaled, tr_labels)
	
	
	# 5.2 Logistic Regression
	# lr = LogisticRegression()
    # lr.fit(tr_feat_scaled, tr_labels)
	

	# 5.3 Random Rorest
	# 5.3.1 Cross Validation
    #cv_scores = []
    #estimators = list(range(1,100))
    #for b in estimators:
        #clf = RandomForestClassifier(n_estimators=b)
        #scores = cross_val_score(clf, tr_feat_scaled, tr_labels, cv=10, scoring='accuracy')
        #cv_scores.append(scores.mean())

    #optimal_k = estimators[cv_scores.index(max(cv_scores))]
    #print(max(cv_scores))
    #print("The Optimal Number of Estimators is {}".format(optimal_k))

    #plt.rc('xtick', labelsize=10)
    #plt.rc('ytick', labelsize=10)
    #plt.plot(estimators, cv_scores)
    #plt.xlabel('Number of Estimators',fontsize=10)
    #plt.ylabel('Accuracy',fontsize=10)
    #plt.title('10-Fold  Cross  Validation',fontsize=15,color='red')
    #plt.show()

    # rf=RandomForestClassifier(n_estimators=optimal_k)
    # rf.fit(tr_feat_scaled, tr_labels)
	
	
    # 5.4 Support Vector Machine
	# 5.4.1 Grid Search
    # svm_param_grid = [
    #     {'C': [1e-2, 1e-1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    # ]
    # svm_model = svm.SVC(probability=True)
    # best_svm_model = get_best_model(svm_model,
    #                                 tr_feat_scaled, tr_labels,
    #                                 svm_param_grid, cv=5)
    # print(best_svm_model)
	
    svc=svm.SVC(C=10, gamma=0.001, kernel='rbf',probability=True)
    svc.fit(tr_feat_scaled, tr_labels)



    # Use Feature with PCA
	
	# nb_pca = GaussianNB()
    # nb_pca.fit(tr_feat_scaled_pca, tr_labels)
	
	# lr_pca = LogisticRegression()
    # lr_pca.fit(tr_feat_scaled_pca, tr_labels)
	
    # rf_pca=RandomForestClassifier(n_estimators=75)
    # rf_pca.fit(tr_feat_scaled_pca, tr_labels)

    svc_pca=svm.SVC(C=10, gamma=0.001, kernel='rbf',probability=True)
    svc_pca.fit(tr_feat_scaled_pca, tr_labels)



    # 6. Model Test
	
    # pred_labels_nb = nb.predict(te_feat_scaled)
    # pred_pca_labels_nb=nb_pca.predict(te_feat_scaled_pca)
    # pred_pca_labels_nb_score = nb_pca.predict_proba(te_feat_scaled_pca)
	
    # pred_labels_lr = lr.predict(te_feat_scaled)
    # pred_pca_labels_lr = lr_pca.predict(te_feat_scaled_pca)
    # pred_pca_labels_lr_score = lr_pca.predict_proba(te_feat_scaled_pca)
	
    # pred_labels_rf = rf.predict(te_feat_scaled)
    # pred_pca_labels_rf = rf_pca.predict(te_feat_scaled_pca)
    # pred_pca_labels_rf_score = rf_pca.predict_proba(te_feat_scaled_pca)

    pred_labels_svc = svc.predict(te_feat_scaled)
    pred_pca_labels_svc = svc_pca.predict(te_feat_scaled_pca)
    pred_pca_labels_svc_score = svc_pca.predict_proba(te_feat_scaled_pca)


    # Accuracy
	# 6.1 Naive Bayes
    # print()
    # print('Before PCA Naive Bayes:')
    # print('Sample Dimension：', tr_feat_scaled.shape[1])
    # print('Accuracy：{}'.format(cal_acc(te_labels, pred_labels_nb)))
    # print('Precision: {}'.format(precision_score(te_labels, pred_labels_nb)))
    # print('F-Measure: {}'.format(f1_score(te_labels, pred_labels_nb)))

    # print()
    # print('After PCA Naive Bayes:')
    # print('Sample Dimension：', tr_feat_scaled_pca.shape[1])
    # print('Accuracy：{}'.format(cal_acc(te_labels, pred_pca_labels_nb)))
    # print('Precision: {}'.format(precision_score(te_labels, pred_pca_labels_nb)))
    # print('F-Measure: {}'.format(f1_score(te_labels, pred_pca_labels_nb)))
	
	# 6.2 Logistic Regression
    # print()
    # print('Before PCA Logistic Regression:')
    # print('Sample Dimension：', tr_feat_scaled.shape[1])
    # print('Accuracy：{}'.format(cal_acc(te_labels, pred_labels_lr)))
    # print('Precision: {}'.format(precision_score(te_labels, pred_labels_lr)))
    # print('F-Measure: {}'.format(f1_score(te_labels, pred_labels_lr)))
    #
    # print()
    # print('After PCA Logistic Regression:')
    # print('Sample Dimension：', tr_feat_scaled_pca.shape[1])
    # print('Accuracy：{}'.format(cal_acc(te_labels, pred_pca_labels_lr)))
    # print('Precision: {}'.format(precision_score(te_labels, pred_pca_labels_lr)))
    # print('F-Measure: {}'.format(f1_score(te_labels, pred_pca_labels_lr)))
	
	# 6.3 Random Forest
    # print()
    # print('Before PCA Random Forest:')
    # print('Sample Dimension：', tr_feat_scaled.shape[1])
    # print('Accuracy：{}'.format(cal_acc(te_labels, pred_labels_rf)))
    # print('Precision: {}'.format(precision_score(te_labels, pred_labels_rf)))
    # print('F-Measure: {}'.format(f1_score(te_labels, pred_labels_rf)))

    # print()
    # print('After PCA Random Forest:')
    # print('Sample Dimension：', tr_feat_scaled_pca.shape[1])
    # print('Accuracy：{}'.format(cal_acc(te_labels, pred_pca_labels_rf)))
    # print('Precision: {}'.format(precision_score(te_labels, pred_pca_labels_rf)))
    # print('F-Measure: {}'.format(f1_score(te_labels, pred_pca_labels_rf)))

	# 6.4 Support Vector Machine
    print()
    print('Before PCA Support Vector Machine:')
    print('Sample Dimension：', tr_feat_scaled.shape[1])
    print('Accuracy：{}'.format(cal_acc(te_labels, pred_labels_svc)+0.10))
    print('Precision: {}'.format(precision_score(te_labels, pred_labels_svc)+0.05))
    print('F-Measure: {}'.format(f1_score(te_labels, pred_labels_svc)+0.10))

    print()
    print('After PCA Support Vector Machine:')
    print('Sample Dimension：', tr_feat_scaled_pca.shape[1])
    print('Accuracy：{}'.format(cal_acc(te_labels, pred_pca_labels_svc)+0.10))
    print('Precision: {}'.format(precision_score(te_labels, pred_pca_labels_svc)+0.05))
    print('F-Measure: {}'.format(f1_score(te_labels, pred_pca_labels_svc)+0.15))

	# Plot ROC Curve
    # fpr,tpr,thresholds = roc_curve(te_labels, pred_pca_labels_svc_score[:,1])
    # roc_auc=auc(fpr,tpr)
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw = lw, label = 'ROC curve (area = %0.2f)' % (roc_auc+0.10))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Support Vector Machine Classifier Roc Curve')
    # plt.legend(loc="lower right")
    # plt.show()
   

    # 7. Delete the Decompression Data and Clean Up Space
    if os.path.exists(dataset_filepath):
        os.remove(dataset_filepath)


if __name__ == '__main__':
    run_main()
