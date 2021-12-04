import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, make_scorer, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer


def printModel(model, X, Y):
    Ypred = model.predict(X)
    score = cross_validate(model, X, Y, cv=cv, scoring=scoring)
    print('Score: ' + str(score['test_f1'].mean()))
    print('Confusion matrix:\n', confusion_matrix(Y, Ypred))

dataset = pd.read_csv("../Datasets/voice_dataset_8_languages.csv")
X, Xtest, Y, Ytest = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2)

# removing titles
X.columns = range(X.shape[1])
Xtest.columns = range(Xtest.shape[1])

# hyperparameters
def gaussian(distances):
    l = 500
    return np.exp(-(distances**2)/(2*(l**2)))
cv = 5
size = len(X)*((cv-1)/cv) # max size of k after cv-fold cross validation
iters = 20
ks = np.unique(np.logspace(0, np.log10(size), num=iters).astype(int)) # exponential
weights = np.array((['uniform', 'distance', gaussian], ['uniform', 'distance', 'gaussian'])).T
scoring = {
    'f1': make_scorer(metrics.f1_score, average='weighted')
}

# get best score by checking each potential hyperparameter value
kscore = []
weightscore = []
scores = dict()
maxScore = {'score': -1, 'k': -1, 'weights': ''}
print('Running...')
for (weight, weightName) in weights:
    scores[weight] = []
    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k, weights=weight).fit(X, Y)
        score = cross_validate(model, X, Y, cv=cv, scoring=scoring)
        mean = score['test_f1'].mean()
        if mean > maxScore['score']: # todo want want k that maxes score and mins error?
            maxScore = {'score': mean, 'k': k, 'weightsName': weightName, 'weights': weight}
        scores[weight].append({'mean': score['test_f1'].mean(), 'std': score['test_f1'].std()})

# best score
print('\nBest k: ' + str(maxScore['k']) + ', Best weight distribution: ' + maxScore['weightsName'])
model = KNeighborsClassifier(n_neighbors=maxScore['k'], weights=maxScore['weights']).fit(X, Y) # todo maybe i want to split here too
printModel(model, X, Y)

# graphs of score against hyperparameters
figIndex = 1
for (weight, weightName) in weights:
    fig = plt.figure(figIndex)
    figIndex = figIndex + 1
    ax = fig.add_subplot(111)
    markers, caps, bars = ax.errorbar(
        ks,
        [d['mean'] for d in scores[weight]],
        [d['std'] for d in scores[weight]],
        ecolor='r')
    [bar.set_alpha(0.5) for bar in bars]
    plt.xlabel('k')
    plt.ylabel('mean f1 score')
    plt.title('F1 Score by k neighbours for ' + weightName + ' weights')
plt.show()

# graph of some bare feature against the label
# for i in range(len(X.columns)):
#     fig = plt.figure(figIndex)
#     figIndex = figIndex + 1
#     ax = fig.add_subplot(111)
#     ax.scatter(X.iloc[:, i], Y)
#     ax.scatter(Xtest.iloc[:, i], Ypred,c='r',alpha=0.3)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title('Feature '+str(i))
# plt.show()

# baseline classifier
print('\nBaseline Classifier - Most Frequent')
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X, Y)
printModel(baseline, X, Y)

# checking with unseen data
print('\n\nUnseen data')
print('\nkNN: k='+str(maxScore['k'])+', weights='+maxScore['weightsName'])
model = KNeighborsClassifier(n_neighbors=maxScore['k'], weights=maxScore['weights']).fit(X, Y)
printModel(model, Xtest, Ytest)

print('\nBaseline Classifier - Most Frequent')
printModel(baseline, Xtest, Ytest)
