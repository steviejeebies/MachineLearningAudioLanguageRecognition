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

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold

# from sklearn.metrics import roc_curve

# For naming any files we may need to export
from datetime import datetime
import os

# import dataset
data_set = pd.read_csv("../Datasets/initial_voice_dataset.csv", header=0, dtype="float")
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
criterion = 'entropy'      
splitter = 'best'
max_depth = None            # 4 seems to be the minimum value that gets perfect scores
min_samples_split = 2
min_samples_leaf = 1
max_features = None         # Not sure how this parameter chooses the features used, 
                            # I suspect it's related to pruning the tree
class_weight = 'balanced'   # None (all outputclasses have equal weight) or 'balanced' 
                            # (ensures the weights of the ouput classes is proportional 
                            # to their frequency in the y list)
ccp_alpha = 0.0             # Using default value for the moment (0.0), related to pruning


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
scores = cross_validate(decision_tree_classifier, feature_data, label_data, cv=5, scoring=scoring)
print(scores)

# strategy = 'constant'
# print("\n\nDUMMY RESULTS ({}):".format(strategy))

# dummy_classifier = DummyClassifier(strategy=strategy)
# scores = cross_validate(dummy_classifier, feature_data, label_data, cv=5, scoring=scoring)

# print(scores)

decision_tree_classifier.fit(feature_data, label_data)

p = plot_tree(decision_tree_classifier, 
                feature_names=feature_data.columns,
                filled=True,
                fontsize=8
            )
plt.show()

# import graphviz
# # DOT data
# dot_data = export_graphviz(decision_tree_classifier, out_file=None, 
#                                 feature_names=feature_data.columns,
#                                 filled=True)

# # Draw graph
# graph = graphviz.Source(dot_data, format="png") 
# graph.render("decision_tree_graphivz")
