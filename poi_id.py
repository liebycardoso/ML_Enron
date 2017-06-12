#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")


# models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Feature normalization.
from sklearn.preprocessing import  MinMaxScaler

# Model selection
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

# Support
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from support_methods import get_score, kbest_scores


### Task 1: Select what features you'll use.

target_label = 'poi'                

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
]

features_list = [target_label] + financial_features_list                  

                 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# To see how I've decided which records to create and clean, check: 
outliers = ["TOTAL", "THE TRAVEL AGENCY IN THE PARK", "LOCKHART EUGENE E", "CHAN RONNIE"]

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
# Instead of using all the features of email, I will summarize in two 
# new attributes

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

# Get Anova e chi2 score for each feature
kbest_scores(features_train, labels_train, features_list)


### Task 4: Try a varity of classifiers

models = []
# naive_bayes
models.append(('Naive Bayes', GaussianNB()))
# tree
models.append(('DecisionTree', DecisionTreeClassifier(random_state=42)))
# ensemble
models.append(('RandomForest', 
               RandomForestClassifier(n_estimators=100, random_state=42)))

models.append(('GradientBoostingClassifier', 
               GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)))

models.append(('AdaBoost', 
               AdaBoostClassifier(n_estimators=100, random_state=42)))

models.append(('ExtraTreesClassifier', 
               ExtraTreesClassifier(n_estimators=100, random_state=42)))
# linear_model
models.append(('LogisticRegression', 
               LogisticRegression(random_state=42)))
# neighbors
models.append(('KNeighbors', 
               KNeighborsClassifier(n_neighbors=5)))

models.append(('NearestCentroid', 
               NearestCentroid()))

# Evaluate each model - default parameters
print get_score(my_dataset, features_list, models)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 

### To see how I've decide for this classifier, check: 

#  Set up several steps to be cross-validated together
pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler(feature_range=(0, 1))),
                           ('classifier', AdaBoostClassifier(random_state=42))
                               ])

# Grid of parameter values
params = {          
          'classifier__base_estimator' : [DecisionTreeClassifier(
                                              class_weight='balanced', 
                                              criterion='gini',
                                              max_depth=None, 
                                              max_features=2, 
                                              max_leaf_nodes=None,
                                              min_samples_leaf=1,
                                              min_samples_split=0.5, 
                                              min_weight_fraction_leaf=0.0,
                                              presort=False, 
                                              random_state=42, 
                                              splitter='random')
                                          ], 
          'classifier__n_estimators': [150,200],          
          'classifier__learning_rate' :[0.01, 0.05, 0.1, 1.0],
          'classifier__algorithm' : ['SAMME.R', 'SAMME']
         }

#  Search for best parameter values for an estimator
grid = GridSearchCV(pipeline, param_grid = params,scoring = 'recall', cv=10)
grid.fit(features_train, labels_train)

clf = grid.best_estimator_

# Test classifier perfomance. (Recall, Precision and Acurracy)
test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. 


dump_classifier_and_data(clf, my_dataset, features_list)


