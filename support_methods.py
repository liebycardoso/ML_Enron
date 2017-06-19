# -*- coding: utf-8 -*-
"""
Created on Sat May 20 21:48:23 2017

@author: lieby
"""
import pandas as pd
from time import time
from sklearn.feature_selection import chi2, f_classif , SelectKBest
from sklearn import model_selection
from sklearn.cross_validation import cross_val_score
#from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import  MaxAbsScaler, StandardScaler, MinMaxScaler
from tester import  test_classifier

def get_model_default_performance(dataset, feature_list, models):
    """
    Estimate the score value by cross-validation for each model.
    Show the result as a DataFrame compose for: Model name, scores, Scores mean 
    
    Args:
        features: Array-like. Data to fit.
        labels:  Array-like. Target variable.
        models: Dictionary. Model name and classifier object.
        name_score: string. Scoring parameter. Example: F1, Accuracy, 
                                                        Recall and precision.    
    """
    
    accuracy_model = []
    for name, clf in models:  
        tm = time()
        print "Start testing classifier:", name
        
        score = test_classifier(clf, dataset, feature_list)
            
        accuracy_model.append([name,score[2], score[3], score[1]])
        print "Successfully completed classifier test."
        print "Processing time: {0}".format(round(time()-tm, 3))
      
    scores = pd.DataFrame(accuracy_model,
                          columns=('Model', 
                                   'Precision', 
                                   'Recall',
                                   'Accuracy')).sort_values(by='Accuracy',
                                                        ascending = False)
    
    
    return scores


def kbest_scores(features, labels, features_list):
    """
    Scores of all features, used for support the parameter search task.
    For each feature return 3 kbest score based on:
        1) Scaled (MinMaxScaler) features
        2) The default function f_classif
        3) Use chi2 function on scaled data
    
    Args:
        features: Array-like. Data to fit.
        labels:  Array-like. Target variable.
        features_list: Array-like. Features names.    
    """
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    selector = SelectKBest(chi2, k='all').fit(features_scaled, labels)
    k_best = SelectKBest(f_classif,k='all').fit(features, labels)    
    
    # Format values
    kbest_pd = pd.DataFrame(zip(features_list[1:],
                                k_best.scores_, # K best score                            
                                selector.scores_), # chi2
                            columns = ['feature','anova', 'chi2'])
    print "SelectKBest: "
    print kbest_pd.sort_values(by='anova',ascending = False)
    
