# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 08:07:01 2017

@author: Chetan
"""

# This code divides data into two parts - one with outOf values from 5 to 16, and other for 17 and up. 
# For predictions on outOf values less than 5, the 5 to 16 predictor is used

import gzip
import numpy as np
import string
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from time import strptime
from textstat.textstat import textstat 

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
    
def calculate_mae_scalar(y_test, pred):
    mae = 0.0
    for i in range(len(y_test)):
        mae += np.absolute(y_test[i] - pred)
    return mae/(len(y_test))

def calculate_mae(y_test, pred):
    mae = 0.0
    for i in range(len(y_test)):
        mae += np.absolute(y_test[i] - (pred[i]))
    return mae/(len(y_test))

def calculate_mse_scalar(y_test, pred):
    mse = 0.0
    for i in range(len(y_test)):
        val = (y_test[i] - pred)
        mse += (val * val)
    return mse/(len(y_test))

def getARI(review):
    return 4.71*(len(review)/review.count(' ') + 1) + 0.5*((review.count(' ') + 1)/(review.count('.')+1))-21.43

allRatings = []
alpha = []
alpha_valid = []
n_helpful = []
out_of = []
words_in_review = []
categories = [1, 2, 3, 4, 5]
price_list = {}
category_wise_price = [0.0,0.0,0.0,0.0,0.0]
category_wise_items = [0,0,0,0,0]
users_rev_cnt = {}
for l in readGz("train.json.gz"):
    if ('price' in l):
        price = l['price']        
        item = l['itemID']
        category = l['categoryID']
        if item not in price_list:
            price_list[item] = price
            category_wise_price[category] += price
            category_wise_items[category] += 1
    user = l['reviewerID']
    if (user in users_rev_cnt):
        count1 = users_rev_cnt[user]
        users_rev_cnt[user] = count1+1 
    else:
        users_rev_cnt[user] = 1
  
category_wise_avg = [0.0,0.0,0.0,0.0,0.0]
for i in range(len(category_wise_items)):
    category_wise_avg[i] = category_wise_price[i]/category_wise_items[i]


# == 0      -> X = 136984
# == 1      -> X = 28551
# >1 && < 9 -> X = 28398
# >= 9      -> X = 6067 
# >=1 and < 9 -> X = 56949
X_1 = np.zeros(shape=(5807, 6), dtype=float) #from 5 to 16
X_2 = np.zeros(shape=(2566, 6), dtype=float)#>=17 -> 1868

y_1 = []
y_2 = []

y_1_helpful = []
y_2_helpful = []

out_of_1 = [1 for i in range(len(X_1))]
out_of_2_8 = []
out_of_9 = []

index_1 = 0
index_2 = 0

overall_cnt = 0
for l in readGz("train.json.gz"):
    if (l['helpful']['outOf'] > 6):# and l['helpful']['outOf'] < 200):
        user,item = l['reviewerID'],l['itemID']
        n_helpful.append(l['helpful']['nHelpful'])
        out_of.append(l['helpful']['outOf'])
        rating = l['rating']
        summary = l['summary']
        review = l['reviewText']
        category = int(l['categoryID'])
        words_in_review.append(review.count(' ') + 1)
        review_time1 = l['reviewTime'].split(',')
        review_time = int(review_time1[1].lstrip())-1996
        month = strptime(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(l['unixReviewTime'])).split(' ')[2],'%b').tm_mon
        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        allRatings.append(rating)
        #score1 = SentimentIntensityAnalyzer().polarity_scores(review)
        #score2 = SentimentIntensityAnalyzer().polarity_scores(summary)
        X = np.zeros(shape=(1,6), dtype=float)
        X[0][0] = l['helpful']['outOf'] #Out of value
        X[0][1] = l['rating'] #Rating value
        if (item in price_list):
            X[0][2] = price_list[item]
        else:
            X[0][2] = category_wise_avg[int(category)]
        X[0][3] = np.log(len(review))
        X[0][4] = textstat.automated_readability_index(review)
        if (user in users_rev_cnt):
            X[0][5] = (users_rev_cnt[user])**2
        if (l['helpful']['outOf'] <= 16):
            X_1[index_1] = X[0]
            y_1.append(l['helpful']['nHelpful']/l['helpful']['outOf'])
            y_1_helpful.append(l['helpful']['nHelpful'])
            index_1 += 1
        else:
            X_2[index_2] = X[0]
            y_2.append(l['helpful']['nHelpful']/l['helpful']['outOf'])
            y_2_helpful.append(l['helpful']['nHelpful'])
            index_2 += 1
        overall_cnt += 1
print ("Finished Reading the data")

########## Trainer for first portion of data #####################
poly_1 = PolynomialFeatures(2)#, interaction_only = True)
X_1 = poly_1.fit_transform(X_1)
#pca_1 = PCA(n_components=29)
#X_1 = pca_1.fit_transform(X_1)

X_1_train, X_1_valid, y_1_train, y_1_valid = train_test_split(X_1, y_1, 
                                                              test_size = 0.25,
                                                              random_state=1)

lamda_1 = 1.0
least_mae_1 = 100000.0
best_lambda_1 = 0
n_estimators = 36   
while lamda_1 < 10:
    reg_lr = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=2,
                                       criterion ='mae')
    reg_lr.fit(X_1_train, y_1_train)
    pred_valid = reg_lr.predict(X_1_valid)
    mae = calculate_mae(y_1_valid, pred_valid)
    if (mae < least_mae_1):
        least_mae_1 = mae
        best_lambda_1 = lamda_1
    print ("MAE = " + str(mae) + " at lambda = " + str(lamda_1))
    #print ("MAE = " + str(mae) + " at estimators = " + str(n_estimators))
    if (lamda_1 < 1):
        lamda_1 += 0.05
    else:
        lamda_1 += 0.5
    n_estimators += 1

print ("Best lambda = " + str(best_lambda_1))
print ("Least MAE = " + str(least_mae_1))
print ("------------------------------")

#Train on Full Data for submission

reg_lr = GradientBoostingRegressor(n_estimators=50, max_depth=3,
                                       criterion ='mae')
reg_lr.fit(X_1, y_1)


########## Training for second batch of data ##################

poly_2 = PolynomialFeatures(2)#, interaction_only = True)
#pca_2 = PCA(n_components=30)
X_2 = poly_2.fit_transform(X_2)
#X_2 = pca_9.fit_transform(X_2)

X_2_train, X_2_valid, y_2_train, y_2_valid = train_test_split(X_2, 
                                                              y_2, 
                                                              test_size = 0.25,
                                                              random_state=5)
lamda_2 = 1.0
n_estimators = 27
least_mae_2 = 100000.0
best_lambda_2 = 0
while lamda_2 <= 10:
#    reg_2 = linear_model.Ridge (alpha = lamda_9, max_iter = 5000, 
#                              random_state=5, normalize=True, 
#                              copy_X=True)
    #reg_2 = AdaBoostRegressor(random_state=5, n_estimators = n_estimators,
    #                          learning_rate=0.01)
    reg_2 = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1,
                                       criterion ='mae')
    reg_2.fit(X_2_train, y_2_train)
    pred_valid = reg_2.predict(X_2_valid)
    for i in range(len(pred_valid)):
        if (pred_valid[i] > 1):
            pred_valid[i] = 1
    mae = calculate_mae(y_2_valid, pred_valid)
    if (mae < least_mae_2):
        least_mae_2 = mae
        best_lambda_2 = lamda_2
    print ("MAE = " + str(mae) + " at lambda = " + str(lamda_2))
    #print ("MAE = " + str(mae) + " at estimators = " + str(n_estimators))
    if (lamda_2 < 1):
        lamda_2 += 0.05
    else:
        lamda_2 += 0.5
    #n_estimators += 1
print ("Best lambda for Ridge Regression at 9 and above = " + str(best_lambda_2))
print ("Least MAE = " + str(least_mae_2))

reg_2 = GradientBoostingRegressor(n_estimators=34, max_depth=2,
                                       criterion ='mae')
reg_2.fit(X_2, y_2)

########### All models trained. Now test on data and write submissions to file################
out_of_test = {}
prediction_dict = {}
known_dict={}
for l in readGz("test_Category.json.gz"):
    known_dict[l['reviewHash']] = l['helpful']['nHelpful']
for l in readGz("train.json.gz"):
    known_dict[l['reviewHash']] = l['helpful']['nHelpful']
    
for l in readGz("test_Helpful.json.gz"):
    user,item = l['reviewerID'],l['itemID']
    out_of1 = l['helpful']['outOf']
    rating = l['rating']
    review = l['reviewText']
    user,item = l['reviewerID'],l['itemID']
    rating = l['rating']
    allRatings.append(rating)
    review = l['reviewText']
    review_time1 = l['reviewTime'].split(',')
    review_time = int(review_time1[1].lstrip())-1996
    summary = l['summary']
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    a_punct = count(review, string.punctuation)
    upperCase = sum(1 for c in review if c.isupper())
    category = int(l['categoryID'])
    #score1 = SentimentIntensityAnalyzer().polarity_scores(review)
    #score2 = SentimentIntensityAnalyzer().polarity_scores(summary)
    month = strptime(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(l['unixReviewTime'])).split(' ')[2],'%b').tm_mon
            
    X_test = np.zeros(shape=(1,6), dtype=float)
    if (out_of1 == 0):
        prediction_dict[user+item] = 0
    elif l['reviewHash'] in known_dict and known_dict[l['reviewHash']] <= out_of1:
        prediction_dict[user+item] = known_dict[l['reviewHash']]
    else:
        X_test[0][0] = l['helpful']['outOf'] #Out of value
        X_test[0][1] = l['rating'] #Rating value
        if (item in price_list):
            X_test[0][2] = price_list[item]
        elif ('price' in l):
            X_test[0][2] = l['price']
            price_list[item] = l['price']
            category_wise_price[int(category)] += price
            category_wise_items[int(category)] += 1
            category_wise_avg = [i/j for (i,j) in zip(category_wise_price, 
                                     category_wise_items)]
        else:
            X_test[0][2] = category_wise_avg[int(category)]
        X_test[0][3] = np.log(len(review))
        X_test[0][4] = textstat.automated_readability_index(review)
        if (user in users_rev_cnt):
            X_test[0][5] = (users_rev_cnt[user])**2
        prediction = [0.5]
        if (out_of1 <= 16):
            X_test = poly_1.transform(X_test)
            prediction = reg_lr.predict(X_test)
        else:#if out_of1 <= 16:
            X_test = poly_2.transform(X_test)
            prediction = reg_2.predict(X_test) 
        prediction[0] = prediction[0] * out_of1
        pred = round(prediction[0])
        prediction_dict[user+item] = pred
        
predictions = open("predictions_Helpful.txt", 'w')
k = 0
for l in open("pairs_Helpful.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i,outOf = l.strip().split('-')
  outOf = int(outOf)
  predictions.write(u + '-' + i + '-' + str(outOf) + ',' 
                      + str(prediction_dict[u+i]) + '\n')
  k += 1

predictions.close()
