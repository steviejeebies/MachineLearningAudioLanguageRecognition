import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, make_scorer

dataset = pd.read_csv("../Datasets/voice_dataset_2.csv")
X, Xtest, Y, Ytest = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2, random_state=0)

# removing titles
X.columns = range(X.shape[1])
Xtest.columns = range(Xtest.shape[1])

for k in [1, 5, 10, 30]:
    model = KNeighborsClassifier(n_neighbors=k).fit(X, Y)
    Ypred = model.predict(X)
    print("k="+str(k))
    print('Confusion matrix:\n', confusion_matrix(Y, Ypred))

    average_param = 'weighted'
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(metrics.precision_score, average=average_param), # todo do we need all of these?
        'recall': make_scorer(metrics.recall_score, average=average_param),
        'f1': make_scorer(metrics.f1_score, average=average_param)
    }
    print("Score:")
    scores = cross_validate(model, X, Y, cv=5, scoring=scoring)
    print(scores['test_f1'])
    print()

# for i in range(len(X.columns)):
#     fig = plt.figure(i)
#     ax = fig.add_subplot(111)
#     ax.scatter(X.iloc[:, i], Y)
#     ax.scatter(Xtest.iloc[:, i], Ypred,c='r',alpha=0.3)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title('Feature '+str(i))
# plt.show()