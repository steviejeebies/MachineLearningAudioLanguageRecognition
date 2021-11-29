## Sources
## https://towardsdatascience.com/decision-tree-algorithm-for-multiclass-problems-using-python-6b0ec1183bf5
## https://www.rockyourcode.com/write-your-own-cross-validation-function-with-make-scorer-in-scikit-learn/
## https://mljar.com/blog/visualize-decision-tree/

### I've separated all the parts of model creation into different sections. If you want to run a section
### this run, then just uncomment it. The only code that is commented-out because it will cause problems
### comes with a warning, you won't miss it.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier

# Cross Validation / Evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import make_scorer
import sklearn.metrics as metrics

from sklearn.preprocessing import PolynomialFeatures

# For naming any files we may need to export
from datetime import datetime
import os

import pprint
pp = pprint.PrettyPrinter(indent=4)

classes=[1,2,3,4,5,6,7,8]       # Replace this with the name of the languages
num_classes = len(classes)

def showAndSave(filename, plt):
    import os
    filename = os.path.dirname(__file__) + '\\images\\' + filename + '_' + str(datetime.now()).replace(':', '-').split('.')[0]
    plt.savefig("{}.png".format(filename), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None)
    plt.show()

def saveCrossEvalAUCResults(this_score_dict, x_axis_values, x_axis_graph_string) :
    plt.errorbar(x_axis_values,this_score_dict['auc']['mean'],yerr=this_score_dict['auc']['std'])
    plt.xlabel(x_axis_graph_string); plt.ylabel('AUC Value')
    max_val = max(this_score_dict['auc']['mean'])
    max_index = this_score_dict['auc']['mean'].index(max_val)
    index_cross_eval_value = x_axis_values[max_index]
    print("HIGHEST AUC VAL = {} AT CROSS_EVAL_VALUE = {}".format(max_val, index_cross_eval_value))
    pp.pprint(this_score_dict)
    showAndSave(x_axis_graph_string, plt)

def fresh_score_dict():
    return {
        'accuracy' : {
            'mean': [],
            'std': [],
        },
        'recall' : {
            'mean': [],
            'std': [],
        },
        'precision' : {
            'mean':[],
            'std': [],
        },
        'f1' : {
            'mean': [],
            'std': [],
        },
        'auc' : {
            'mean': [],
            'std': [],
        }
    }

################# DUMMY CLASSIFIER #################
## Although the different strategies for DummyClassifiers take different approaches
## for how they make their predictions, all the strategies produced identical results
## when modelled on this dataset (with the exception of stratefied, which performed
# very slightly better on some classes and very slightly worse on others, when compared
# with the other dummy classifier strategies).
strategy = 'most_frequent'
dummy_classifier = DummyClassifier(strategy=strategy)


################# GETTING DATA #################

# import dataset
data_set = pd.read_csv("../Datasets/voice_dataset_8_languages.csv", header=0, dtype="float")
feature_data = data_set.iloc[:,:-1]     # Everything except last column
label_data = data_set.iloc[:,-1]        # Just last column (the class label)

# Shuffling the data - if I don't do this, then evaluation results are strange 
# (as high as 0.98, as low as 0.2, for the same metric in different iterations of kFold)
from sklearn.utils import shuffle
feature_data, label_data = shuffle(feature_data, label_data)

################# HYPERPARAMETER OPTIONS #################
# putting them all grouped together here so we don't have to dig through code later
### For DecisionTreeModel constructor parameters (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
criterion = 'gini'      
splitter = 'best'           # 'best' or 'random', best will give consistent results, but 'random' may reveal some important features
max_depth = None            # 4 seems to be the minimum value that gets perfect scores (for initial data set, not tested on the rest)
min_samples_leaf = 1        # A leaf will not be created if it has less than this number of samples in it. 
max_features = None         # Not sure how this parameter chooses the features used, I suspect it's related to pruning the tree
random_state = None         # Used in conjunction with `max_features`. Regardless of the 'splitter' value, features are always randomly permuted at each split. If the criteria for determining which feature to split on is identical for all, then this randomness determines. We can set to an int for deterministic behaviour for this. 
max_leaf_nodes = None       # Tree will have no more than this number of leaf nodes total, best-first fashion. None = unliimited leaf nodes.
class_weight = 'balanced'   # None (all output classes have equal weight) or 'balanced' (ensures the weights of the ouput classes is proportional to their frequency in the y list)
ccp_alpha = 0.0             # Using default value for the moment (0.0). Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. 
poly_feature_num = 1

################# HYPERPARAMETERS WE'RE CONSIDERING #################
#### For Pre-Pruning Approach:
CV_min_samples_leaf = range(1, 30)
CV_max_features = [1, 5, 10, 15, 20, 25, 30, 35, 40, 46]   # 46 features total
CV_max_depth = range(1, 15)
# CV_criterion = ['gini', 'entropy']    # Had no impact
# CV_max_leaf_nodes = range(1, 20)      # This parameter turned out to be useless - anything below 10, terrible results, but anything above 10, all equal. The default for this parameter will use the latter behaviour

#### For Post-Pruning Approach:
## ccp_alpha
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(feature_data, label_data)
CV_ccp_alphas = path.ccp_alphas
CV_ccp_alphas = list(filter(lambda x : x > 0, CV_ccp_alphas))   # filtering out all the ccp_alphas values that were 0
CV_ccp_alphas = CV_ccp_alphas[::5]       # this returns a massive number of values, many are very similar. We'll get a representative subset of these
CV_ccp_alphas = CV_ccp_alphas[:-2]       # Need to remove last two values of the above set, since they produce trees that are far too poor to be useful, and mess up the axis of the graph

#### NOTE: Post-Pruning and Pre-Pruning are distinct, don't mix them up!
## difference between pre-pruning and post-pruning 
#       https://www.educative.io/edpresso/what-is-decision-tree-pruning-and-how-is-it-done
## post-pruning approach 
#       https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

## Polynomial Features, if set to 1, no change from input. Decision trees don't benefit
## from higher polynomial features at all, but I wanted to see if any significant features
## appeared, regardless
pf = PolynomialFeatures(poly_feature_num)
feature_data = pf.fit_transform(feature_data)
feature_names = pf.get_feature_names_out(input_features=data_set.columns[:-1])

# print("""\n
#     PARAMETERS:\n
#     \tcriterion = {}\n
#     \tsplitter = {}\n
#     \tmax_depth = {}\n
#     \tmin_samples_leaf = {}\n
#     \tmax_features = {}\n
#     \trandom_state = {}\n
#     \tmax_leaf_nodes = {}\n
#     \tclass_weight = {}\n
#     \tccp_alpha = {}\n
#     \tpoly_feature_num = {}\n
# """.format(criterion, splitter, max_depth, min_samples_leaf, max_features, random_state, max_leaf_nodes, class_weight, ccp_alpha, poly_feature_num))

################# CROSS VALIDATION APPRAOCH #################
# I will be cross-evaluating each hyperparameter individually (while setting all other values to default),
# But I will also be using the Randomized Search Cross Validation Approach to learn which combination of 
# parameters produces the best model.
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

# Custom scoring dictionary (this was required for multiclass models when using cross_validate)
average_param = 'weighted'
scoring = {'accuracy': 'accuracy',
           'recall': make_scorer(metrics.recall_score, average=average_param),
           'precision': make_scorer(metrics.precision_score, average=average_param),
           'f1': make_scorer(metrics.f1_score, average=average_param),
           'auc': make_scorer(metrics.f1_score, average=average_param),
           }

score_dict = fresh_score_dict()

################# CROSS VALIDATION, EACH HYPERPARAMETER #################
#### Important note;
## In the definition of the for-loop on the next line, you need to replace both the
## variable (that starts with TEST_) and the list (that starts with CV_) with the parameter
## you actually want to check this run. You also need to change the relevant parameter in the 
## decision tree classifier, just below it. Finally, below this for-loop, uncomment
## the relevant function call for this parameter.

# for TEST_ccp_alpha in CV_ccp_alphas:
#     # Creating model:
#     decision_tree_classifier = DecisionTreeClassifier(
#         criterion = criterion,
#         splitter = splitter,
#         max_depth = max_depth,
#         min_samples_leaf = min_samples_leaf,
#         max_features = max_features,
#         random_state = random_state,
#         max_leaf_nodes = max_leaf_nodes,
#         class_weight = class_weight,
#         ccp_alpha = TEST_ccp_alpha
#     )

#     scores = cross_validate(decision_tree_classifier, feature_data, label_data, cv=5, scoring=scoring)

#     score_dict['accuracy']['mean'].append(scores['test_accuracy'].mean())
#     score_dict['accuracy']['std'].append(scores['test_accuracy'].std())

#     score_dict['recall']['mean'].append(scores['test_recall'].mean())
#     score_dict['recall']['std'].append(scores['test_recall'].std())

#     score_dict['precision']['mean'].append(scores['test_precision'].mean())
#     score_dict['precision']['std'].append(scores['test_precision'].std())

#     score_dict['f1']['mean'].append(scores['test_f1'].mean())
#     score_dict['f1']['std'].append(scores['test_f1'].std())

#     score_dict['auc']['mean'].append(scores['test_auc'].mean())
#     score_dict['auc']['std'].append(scores['test_auc'].std())

##### UNCOMMENT THE RELEVANT PARAMETER YOU'RE CHECKING THIS RUN
# saveCrossEvalAUCResults(score_dict, CV_min_samples_leaf, 'Minimum Samples to Leaf')
# saveCrossEvalAUCResults(score_dict, CV_max_features, 'Max Features Used')
# saveCrossEvalAUCResults(score_dict, CV_max_depth, 'Max Depth of Tree')
# saveCrossEvalAUCResults(score_dict, CV_ccp_alphas, 'CCP Alphas Used For Pruning Tree')


################# CROSS VALIDATION, COMBINED HYPERPARAMETERS #################
# ## Gives us the best value for all parameters, when combined.
# ## Base version of Decision Tree
# dt_base = DecisionTreeClassifier()

# dt_parameter_grid = {
#             'min_samples_leaf': CV_min_samples_leaf,
#             'max_features': CV_max_features,
#             'max_depth': CV_max_depth,
#             # 'criterion': CV_criterion,    # little impact
# }

# # Testing values for parameters as defined in dt_parameter_grid. This creates a new model
# random = RandomizedSearchCV(estimator = dt_base, param_distributions = dt_parameter_grid, 
#                                n_iter = 1500, 
#                                cv = 5, 
#                                verbose = 0, 
#                                random_state = 1, 
#                                n_jobs = -1      # run in parallel
#                            )

# random.fit(feature_data, label_data)

# # Get the best parameters produced
# print(random.best_params_)

################# EVALUATION, COMPARISON OF TREES #################

# The following is the Decision Tree that uses the best parameters for pre-pruned trees, determined by a RandomizedSearchCV() run at 500 iterations:
pre_pruned_decision_tree_classifier = DecisionTreeClassifier(
        max_depth = 10,
        min_samples_leaf = 23,
        max_features = 30,
        class_weight = 'balanced'
    )

# The following is the Decision Tree that has the optimal ccp_alpha value, according to cross validation
post_pruned_decision_tree_classifier = DecisionTreeClassifier(
    ccp_alpha = 0.00396237028052095
)

# from sklearn.preprocessing import label_binarize
# binarized_label = label_binarize(label_data, classes=classes)

# # Just considering one split now, we've already done the k-Fold work
# Xtrain, Xtest, ytrain, ytest = train_test_split(feature_data, binarized_label, test_size=0.2)

# # OneVsRestClassifier - necessary for detecting the TP/FP/AUC for each individual class
# prepruning_one_vs_rest = OneVsRestClassifier(pre_pruned_decision_tree_classifier)
# postpruning_one_vs_rest = OneVsRestClassifier(post_pruned_decision_tree_classifier)
# dummy_one_vs_rest = OneVsRestClassifier(dummy_classifier)

# y_score_pre     = prepruning_one_vs_rest.fit(Xtrain, ytrain).predict_proba(Xtest)
# y_score_post    = postpruning_one_vs_rest.fit(Xtrain, ytrain).predict_proba(Xtest)
# y_score_dummy   = dummy_one_vs_rest.fit(Xtrain, ytrain).predict_proba(Xtest)

# cmap = plt.cm.get_cmap('tab10', num_classes)

# graphs_iter = zip(
#     ["Pre-Pruning Decision Tree ROC", "Post-Pruning Decision Tree ROC", "Dummy ROC"], 
#     [y_score_pre, y_score_post, y_score_dummy]
# )

# for graph_name, score in graphs_iter:
#     fp = {}; tp = {}; auc_val = {}
#     for i in range(num_classes):
#         fp[i], tp[i], _ = roc_curve(ytest[:, i], score[:, i])
#         auc_val[i] = auc(fp[i], tp[i])
#     for i in range(num_classes):
#         plt.plot(fp[i], tp[i], color=cmap(i), label='{0} (AUC = {1:0.2f})'.format(classes[i], auc_val[i]))
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Graph for {}'.format(graph_name))
#     plt.legend(loc="lower right")
#     showAndSave(graph_name, plt)

################# CONFUSION MATRICES #################

# # Have to create train/test data again, as last time, the classes were binarized
X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.2)

pre_pruned_decision_tree_classifier.fit(X_train, y_train)
post_pruned_decision_tree_classifier.fit(X_train, y_train)
dummy_classifier.fit(X_train, y_train)

# cm_options = [
#     ("Post-Pruned Decision Tree Confusion Matrix", post_pruned_decision_tree_classifier),
#     ("Pre-Pruned Decision Tree Confusion Matrix", pre_pruned_decision_tree_classifier),
# ]
# for title, classifier in cm_options:
#     disp = ConfusionMatrixDisplay.from_estimator(
#         classifier,
#         X_test,
#         y_test,
#         display_labels=classes,
#         cmap=plt.cm.Blues,
#         normalize='true',
#     )
#     disp.ax_.set_title(title)
#     showAndSave(title, plt)



################# FEATURE IMPORTANCE #################

# print('FOR PRE-PRUNED TREE')
# ## This code was borrowed from https://towardsdatascience.com/decision-tree-algorithm-for-multiclass-problems-using-python-6b0ec1183bf5
# pre_pruned_decision_tree_classifier.tree_.compute_feature_importances(normalize=True)
# feat_imp_dict = dict(zip(feature_names, pre_pruned_decision_tree_classifier.feature_importances_))
# feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
# feat_imp.rename(columns = {0:'FeatureImportance'}, inplace = True)
# print(feat_imp.sort_values(by=['FeatureImportance'], ascending=False).head())

# print('FOR POST-PRUNED TREE')
# post_pruned_decision_tree_classifier.tree_.compute_feature_importances(normalize=True)
# feat_imp_dict = dict(zip(feature_names, post_pruned_decision_tree_classifier.feature_importances_))
# feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
# feat_imp.rename(columns = {0:'FeatureImportance'}, inplace=True)
# print(feat_imp.sort_values(by=['FeatureImportance'], ascending=False).head())

################# GRAPHING DECISION GRAPH #################
## The following requires that Graphviz is installed on your machine and added to the System PATH, 
## just installing the Python module for Graphviz is not enough
## https://graphviz.org/download/

from dtreeviz.trees import dtreeviz

viz1 = dtreeviz(post_pruned_decision_tree_classifier,
                feature_data,
                label_data,
                target_name="language",
                feature_names=feature_names,
                )

viz1.save("./images/pre_pruned_decision_tree.svg")

## NOTE: This causes an error, and will not print the tree. Seems to be an issue with dtreeviz, 
## as this call is identical to the previous, but with a different tree used.

# viz2 = dtreeviz(preFORMERLYpostpruned_decision_tree_classifier,
#                 feature_data,
#                 label_data,
#                 target_name="language",
#                 feature_names=feature_names,
#                 )

# viz2.save("./images/post_pruned_decision_tree.svg")



