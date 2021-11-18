## Sources
## https://towardsdatascience.com/decision-tree-algorithm-for-multiclass-problems-using-python-6b0ec1183bf5
## https://www.rockyourcode.com/write-your-own-cross-validation-function-with-make-scorer-in-scikit-learn/
## https://mljar.com/blog/visualize-decision-tree/


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.dummy import DummyClassifier

# Cross Validation / Evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
import sklearn.metrics as metrics

from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold

# from sklearn.metrics import roc_curve

# For naming any files we may need to export
from datetime import datetime
import os

# import dataset
data_set = pd.read_csv("../Datasets/voice_dataset_2.csv", header=0, dtype="float")
feature_data = data_set.iloc[:,:-1]     # Everything except last column
label_data = data_set.iloc[:,-1]        # Just last column (the class label)

# feature_data = data_set.iloc[:250,:]
# label_data = data_set.iloc[:250,:]

# X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.1, random_state=0)

# print(feature_data.size)
# print(X_train.size)
# print(X_test.size)

# PARAMETERS USED FOR CROSS VALIDATION (putting them
# all grouped together here so we don't have to dig 
# through code later)
### For DecisionTreeModel constructor parameters (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
criterion = 'gini'      
splitter = 'best'           # 'best' or 'random', best will give consistent results, but 'random' may reveal some important features
max_depth = None            # 4 seems to be the minimum value that gets perfect scores (for initial data set, not tested on the rest)
min_samples_split = 5
min_samples_leaf = 1
max_features = None         # Not sure how this parameter chooses the features used, 
                            # I suspect it's related to pruning the tree
class_weight = 'balanced'   # None (all outputclasses have equal weight) or 'balanced' 
                            # (ensures the weights of the ouput classes is proportional 
                            # to their frequency in the y list)
ccp_alpha = 0.0             # Using default value for the moment (0.0), related to pruning
poly_feature_num = 1

print("""\n
    PARAMETERS:\n
    \tcriterion = {}\n
    \tsplitter = {}\n
    \tmax_depth = {}\n
    \tmin_samples_split = {}\n
    \tmin_samples_leaf = {}\n
    \tmax_features = {}\n
    \tclass_weight = {}\n
    \tccp_alpha = {}\n
    \tPolynomial Features = {}\n
""".format(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, class_weight, ccp_alpha, poly_feature_num))

pf = PolynomialFeatures(poly_feature_num)
feature_data = pf.fit_transform(feature_data)
feature_names = pf.get_feature_names_out(input_features=data_set.columns[:-1])

# Creating model:
decision_tree_classifier = DecisionTreeClassifier(
    criterion=criterion,
    splitter=splitter,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    class_weight=class_weight,
    ccp_alpha=ccp_alpha
)

# Shuffling the data - if I don't do this, then evaluation results are strange 
# (as high as 0.98, as low as 0.2, for the same metric in different iterations of kFold)
from sklearn.utils import shuffle
feature_data, label_data = shuffle(feature_data, label_data)

# dt_classifier.fit(X_train, y_train)
# print(dt_classifier.score(X_test, y_test))

# Custom scoring dictionary (this was required for multiclass models when using cross_validate)
average_param = 'weighted'
scoring = {'accuracy': 'accuracy',
           'precision': make_scorer(metrics.precision_score, average=average_param),
           'recall': make_scorer(metrics.recall_score, average=average_param),
           'f1': make_scorer(metrics.f1_score, average=average_param)
           }

print("DECISION TREE RESULTS:")
scores = cross_validate(decision_tree_classifier, feature_data, label_data, cv=10, scoring=scoring)
print(scores)

# strategy = 'constant'
# print("\n\nDUMMY RESULTS ({}):".format(strategy))

# dummy_classifier = DummyClassifier(strategy=strategy)
# scores = cross_validate(dummy_classifier, feature_data, label_data, cv=5, scoring=scoring)

# print(scores)

decision_tree_classifier.fit(feature_data, label_data)

# This code was borrowed from https://towardsdatascience.com/decision-tree-algorithm-for-multiclass-problems-using-python-6b0ec1183bf5
decision_tree_classifier.tree_.compute_feature_importances(normalize=False)
feat_imp_dict = dict(zip(feature_names, decision_tree_classifier.feature_importances_))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp.rename(columns = {0:'FeatureImportance'}, inplace = True)
print(feat_imp.sort_values(by=['FeatureImportance'], ascending=False).head())

# p = plot_tree(decision_tree_classifier, 
#                 feature_names=feature_data.columns,
#                 filled=True,
#                 fontsize=8
#             )
# plt.show()