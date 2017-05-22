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
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from support_methods import get_f1_score, Kbest_scores
from sklearn.preprocessing import  MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.svm import SVC

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
    #'total_stock_value',
]

#features_list = [target_label] + financial_features_list + email_features_list                 
features_list = [target_label] + financial_features_list                  

                 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

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
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,  test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, 
                                                                labels,  
                                                                test_size=0.3, 
                                                                random_state=42)

#get_f1_score(features_train, labels_train)

#Kbest_scores(features_train, labels_train, X_train, y_train, features_list)



#print x
    #print "Model:", name    
    #print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)



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

###

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
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
models.append(('SVM', SVC()))


"""
# prepare configuration for cross validation test harness
seed = 32
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, clf in models:
    
    kfold = model_selection.KFold(n_splits=10)
	 cv_results = model_selection.cross_val_score(model, features_train, labels_train, cv=kfold, scoring=scoring)
	 results.append(cv_results)
	 names.append(name)
	 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	 print(msg)
    
    results.append(test_classifier(clf, my_dataset, features_list))
    
x =  pd.DataFrame(results, columns=['Model', 
                                     'Accuracy', 
                                     'Precision',
                                     'Recall',
                                     'F1',
                                     'F2']).sort_values(by='F1',ascending = False))
"""
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
# models.append(('AdaBoost', AdaBoostClassifier(n_estimators=100)))
# models.append(('NearestCentroid', NearestCentroid()))

param_test1 = {'n_estimators':range(20,81,10)}
grid = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, 
                                                               min_samples_split=500,
                                                               min_samples_leaf=50,
                                                               max_depth=8,
                                                               max_features='sqrt',
                                                               subsample=0.8,
                                                               random_state=10), 
                    param_grid = param_test1, 
                    scoring='roc_auc',n_jobs=4,iid=False, cv=5)
grid.fit(features_train, labels_train)
grid.best_estimator_
# predicted = grid.predict(features_test)
# x = classification_report(labels_test, predicted)
#test_classifier(grid.best_estimator_, my_dataset, features_list)    
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)
