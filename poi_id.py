#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn import cross_validation
from time import time
import pandas as pd
import pylab as pl
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from support_methods import get_score, kbest_scores
from sklearn.preprocessing import  MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit

def get_recall_precision(label, prediction): 

    precision = precision_score(label, prediction, labels=None, pos_label=1, average='binary', sample_weight=None)
    print precision
    
    recall = recall_score(label, prediction, labels=None, pos_label=1, average='binary', sample_weight=None)
    return recall , precision

    


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

target_label = 'poi'                
email_features_list = [
    # 'email_address', # remit email address; informational label
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]

financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary'
    #,
    #'total_payments',
    #'total_stock_value'
]

#features_list = [target_label] + financial_features_list + email_features_list                 
features_list = [target_label] + financial_features_list                  

                 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
outliers = ["TOTAL", "THE TRAVEL AGENCY IN THE PARK", "LOCKHART EUGENE E", "CHAN RONNIE"]
#outliers = ["TOTAL", "THE TRAVEL AGENCY IN THE PARK"]
for outlier in outliers :
    data_dict.pop(outlier, 0)


def update_dict_value(key, items, values, dict_obj):
    index = 0
    for item in items:     
        dict_obj[key][item] = values[index]
        index += 1
    return dict_obj
        
 
data_dict = update_dict_value(
              'BELFER ROBERT',
              ['deferred_income','deferral_payments', 'expenses', 
               'director_fees', 'total_payments', 'exercised_stock_options',
               'restricted_stock','restricted_stock_deferred',
               'total_stock_value'], 
              [-102500,'NaN',3285,102500, 3285,'NaN', 44093,-44093,'NaN'],
              data_dict)


data_dict = update_dict_value(
              'BHATNAGAR SANJAY',
              ['other', 'expenses', 'director_fees', 'total_payments',
               'exercised_stock_options','restricted_stock',
               'restricted_stock_deferred','total_stock_value'],
              ['NaN',137864, 'NaN', 137864, 15456290, 
               2604490, -2604490, 15456290],
               data_dict)

### Task 3: Create new feature(s)
for key in data_dict:
    key_values = data_dict[key]

    total_msg = (data_dict[key]['to_messages'] + 
                 data_dict[key]['from_messages'])
    
    total_poi_msg = (data_dict[key]['from_poi_to_this_person'] +
                     data_dict[key]['from_this_person_to_poi'] + 
                     data_dict[key]['shared_receipt_with_poi'])     
        
    try:
        data_dict[key]['message_poi_ratio'] = (float(total_poi_msg) / 
                                           float(total_msg))
    except:
        data_dict[key]['message_poi_ratio'] = "NaN"
        
    try:
        data_dict[key]['message_others_ratio'] = ((float(total_msg) - float(total_poi_msg)) / 
                                          float(total_msg))
    except:
        data_dict[key]['message_others_ratio'] = "NaN"

features_list = features_list + ['message_poi_ratio','message_others_ratio'] 
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, 
                                                remove_NaN = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,  test_size=0.3, random_state=42)


#kbest_scores(features_train, labels_train, features_list)

###for i in [0,1,2,3,5,6,7,9]:
###    models.pop(i)

###len(models)   



 
"""        
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

"""
###try:
###    prettyPicture(clf, features_test, labels_test)
###except NameError:
###    pass
################################

### Task 4: Try a varity of classifiers

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"

models = []
# naive_bayes
models.append(('Naive Bayes', GaussianNB()))
# tree
models.append(('DecisionTree', DecisionTreeClassifier(random_state=0)))
# ensemble
models.append(('RandomForest', RandomForestClassifier(n_estimators=100, random_state=0)))
models.append(('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)))
models.append(('AdaBoost', AdaBoostClassifier(n_estimators=100)))
models.append(('ExtraTreesClassifier', ExtraTreesClassifier(n_estimators=100, random_state=0)))
# linear_model
models.append(('LogisticRegression', LogisticRegression()))
# neighbors
models.append(('KNeighbors', KNeighborsClassifier(n_neighbors=4)))
models.append(('NearestCentroid', NearestCentroid()))
# SVC
#models.append(('SVM', SVC()))

print get_score(my_dataset, features_list, models)

#print get_score(features_train, labels_train, models, 'precision')
scv = StratifiedShuffleSplit(labels_train, 10, random_state = 42)

"""
seed = 32

results = []

# evaluate each model - basics parameters

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 

# models.append(('AdaBoost', AdaBoostClassifier(n_estimators=100)))
# models.append(('NearestCentroid', NearestCentroid()))


scv = StratifiedShuffleSplit(labels_train, 1000, random_state = 42)

Pipeline(steps=[('minmaxer', MinMaxScaler(copy=True, feature_range=(0, 1))), ('selection', SelectKBest(k=10, score_func=<function f_classif at 0x000000000C5EB9E8>)), ('classifier', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
          n_estimators=200, random_state=None))])
        Accuracy: 0.85560       Precision: 0.44459      Recall: 0.33300 F1: 0.38079     F2: 0.35060
        Total predictions: 15000        True positives:  666    False positives:  832   False negatives: 1334   True negatives: 12168



pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler(feature_range=(0, 1))),
                           ('selection', SelectKBest()),
                           ('classifier', AdaBoostClassifier())
                               ])
params = {'selection__k': [10, 'all'],
          'classifier__n_estimators':[150,200],
          'classifier__learning_rate' :[0.1, 1],
          'classifier__algorithm' : ['SAMME.R', 'SAMME'] 
               }
# set up gridsearch
grid = GridSearchCV(pipeline, param_grid = params,
                          scoring = 'f1', cv =scv)
grid.fit(features, labels)

clf = grid.best_estimator_

test_classifier(clf, my_dataset, features_list)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', 
                     metric_params=None, n_neighbors=5, p=2, weights='distance')

#### RUIM
pipeline = Pipeline(steps=[('minmaxer',MinMaxScaler(feature_range=(0, 1))),
                           ('selection', SelectKBest()),
                           ('pca', PCA(n_components=.95, random_state=42)),
                           ('classifier', KNeighborsClassifier())
                          ])

params = {'selection__k': [ 10,'all'],
          'classifier__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
          'classifier__metric': ['minkowski'],
          'classifier__metric_params'  : [None],
          'classifier__n_neighbors'  : [4,5,6,7],
          'classifier__p' : [1,2],
          'classifier__weights'  : ['distance', 'uniform']
         }

grid = GridSearchCV(pipeline, 
                    param_grid = params, 
                    scoring='accuracy', cv=scv)

grid.fit(features_train, labels_train)

clf = grid.best_estimator_

test_classifier(clf, my_dataset, features_list)

Pipeline(steps=[('minmaxer', MinMaxScaler(copy=True, feature_range=(0, 1))), ('selection', SelectKBest(k=10, score_func=<function f_classif at 0x000000000C5EB9E8>)), ('classifier', NearestCentroid(metric='cityblock', shrink_threshold=None))])
        Accuracy: 0.83567       Precision: 0.36490      Recall: 0.31400 F1: 0.33754     F2: 0.32301
        Total predictions: 15000        True positives:  628    False positives: 1093   False negatives: 1372   True negatives: 11907

pipeline = Pipeline(steps=[('minmaxer', MaxAbsScaler()),
                           ('selection', SelectKBest()),
                           ('classifier', NearestCentroid())
                          ])

params = {'selection__k': [ 6, 10, 14, 'all'],
          'classifier__metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan','correlation', 'minkowski'],
          'classifier__shrink_threshold'  : [None, .2]
         }

grid = GridSearchCV(pipeline, 
                    param_grid = params, 
                    scoring='accuracy', cv=scv)
grid.fit(features_train, labels_train)

scv = StratifiedShuffleSplit(labels_train, 1000, random_state = 42)
pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler(feature_range=(0, 1))),
                           ('selection', SelectKBest()),
                           ('classifier', DecisionTreeClassifier(random_state = 0))
                          ])

params = {'selection__k': [10, 'all'],
          "classifier__criterion": ["gini", "entropy"],
          #'classifier__min_samples_split': [2,4],
          'classifier__max_features': [1,3]
         }

grid = GridSearchCV(pipeline, 
                    param_grid = params, 
                    scoring='accuracy', cv=scv)

grid.fit(features_train, labels_train)

pred = grid.predict(features_test)

print 'Accuracy:', accuracy_score(labels_test, pred)
#print 'F1 score:', f1_score(y_test, prediction)
print 'Recall:', recall_score(labels_test, pred)
print 'Precision:', precision_score(labels_test, pred)
#test_classifier(grid.best_estimator_, my_dataset, features_list)

pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                           ('selection', SelectKBest()),
                           ('classifier', LogisticRegression())
                          ])

params = {'selection__k': [10, 14, 'all'],
          #'classifier__C': [0.05, 0.5, 1, 10, 100, 500, 1000],
          #'classifier__solver': ['liblinear'],
          'classifier__penalty': ['l1', 'l2'], 
          'classifier__C': [0.1, 0.5, 1, 10, 100],
          'classifier__class_weight': [None, 'balanced']
         }

grid = GridSearchCV(pipeline, 
                    param_grid = params, 
                    scoring='recall', cv=scv)

grid.fit(features_train, labels_train)

pred = grid.predict(features_test)

precision, recall, fscore, support = precision_recall_fscore_support(labels_test, pred)

#print('precision: {}'.format(precision))
#print('recall: {}'.format(recall))
#print('fscore: {}'.format(fscore))
#print('support: {}'.format(support))
clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=200, random_state=None)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

test_classifier(clf, my_dataset, features_list)


print 'Accuracy:', accuracy_score(labels_test, pred)
#print 'F1 score:', f1_score(y_test, prediction)
print 'Recall:', recall_score(labels_test, pred)
print 'Precision:', precision_score(labels_test, pred)
#test_classifier(grid.best_estimator_, my_dataset, features_list)

# clf = search.best_estimator_



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. 


#dump_classifier_and_data(clf, my_dataset, features_list)


#http://www.ritchieng.com/machine-learning-project-student-intervention/
#https://github.com/baumanab/udacity_intro_machinelearning_project/blob/master/final_project/my_poi_id.py
"""