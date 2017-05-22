# -*- coding: utf-8 -*-
"""
Created on Sat May 20 21:48:23 2017

@author: lieby
"""
from tester import dump_classifier_and_data, test_classifier
from sklearn import cross_validation
from time import time
import pandas as pd
import pylab as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import chi2, f_regression, SelectKBest
from sklearn import model_selection
# Preprocessing
from sklearn.preprocessing import  MaxAbsScaler, StandardScaler, MinMaxScaler

from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit

def get_f1_score(features, labels):
    classifier = [GaussianNB(), 
                  DecisionTreeClassifier(), 
                  RandomForestClassifier(), 
                  KNeighborsClassifier(n_neighbors=4),
                  AdaBoostClassifier(),
                  LogisticRegression(),
                  NearestCentroid(),
                  GradientBoostingClassifier(),
                  ExtraTreeClassifier()                          
                 ]
    classifier_name = ['Naive Bayes', 
                       'Decision Tree', 
                       'Random Forest',
                       'KNeighbors',
                       'AdaBoost',
                       'Logistic Regression',
                       'NearestCentroid',
                       'GradientBoostingClassifier',
                       'ExtraTreeClassifier']
    
    accuracy_model = []
    for clf, name in zip(classifier, classifier_name):    
        score = cross_val_score(clf, features, labels, scoring='f1')
        accuracy_model.append([name,score[0],score[1],score[2],score.mean()])
      
    scores = pd.DataFrame(accuracy_model,columns=('Model', 
                                                 'Score1', 
                                                 'Score2',
                                                 'Score3',
                                                 'Mean')).sort_values(by='Mean',ascending = False)
    print scores


def Kbest_scores(features, labels,features_scaled, y_train, features_list):
        
    

    selector = SelectKBest(chi2, k='all').fit(features_scaled, y_train)
    #selector = SelectKBest(k='all').fit(features_scaled, y_train)
    # ANOVA F-value between label/feature for classification tasks
    k_best = SelectKBest(f_regression,k='all').fit(features, labels)
    k_best_scaled = SelectKBest(k='all').fit(features_scaled, y_train)
    
    
    # Format values
    kbest_pd = pd.DataFrame(zip(features_list[1:], 
                                k_best_scaled.scores_, # scaled
                                k_best.scores_, # K best score                            
                                selector.scores_), # chi2
                            columns = ['feature','anova_scaled','anova', 'chi2'])
    print "SelectKBest"
    print kbest_pd.sort_values(by='anova_scaled',ascending = False)