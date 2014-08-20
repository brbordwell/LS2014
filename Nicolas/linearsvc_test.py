from sklearn import svm, cross_validation, metrics
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

"""
Small script for testing Linear SVC with cross-validation
"""

NUM_OF_ROWS = 10000



# Select file and load raw data into a matrix
file_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data_csv', 'GalaxyZooData.csv')
#data = np.genfromtxt(file_path, delimiter=',', names=True).transpose()

#file_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data_csv', 'GalaxyZooData_2.csv')
with open(file_path) as t_in:
	data = np.genfromtxt(itertools.islice(t_in, NUM_OF_ROWS+1), delimiter=',', names=True)

# From structured array to nparray
data_svm = data.view(np.float64).reshape(data.shape + (-1,))

# Identify elipticals and spirals
index_el, index_cs, index_objid = data.dtype.names.index('p_el_debiased'), data.dtype.names.index('p_cs_debiased'), data.dtype.names.index('objid')

# Filter the unknowns
elliptical, spiral = data_svm[:, index_el] > .8, data_svm[:, index_cs] > .8
data_svm = data_svm[elliptical | spiral]

# Get classes
spiral = data_svm[:, index_cs] > .8
classes = np.zeros(len(spiral), dtype=bool) | spiral

# Generate a vector of classes
#elliptical, spiral = data['p_el_debiased'] > .8, data['p_cs_debiased'] > .8
#unknown = np.ones(len(elliptical), dtype=bool) & (~elliptical) & (~spiral)
#classes = elliptical * 1 + spiral * 2 + unknown * 3

# Select training data from raw data without classification features... OR ELSE THAT WOULD BE CHEATING!! ;)
#svn_names = [x for x in data.dtype.names if x != "p_el_debiased" and x != "p_cs_debiased" and x != "objid"]
#data_svm = data[svn_names]
data_svm = np.delete(data_svm, 0, 1)
data_svm = np.delete(data_svm, 0, 1)
data_svm = np.delete(data_svm, 0, 1)

clf = svm.SVC(kernel='linear')

X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_svm, classes, test_size=1./3.)
print "training set = ", X_train.shape, y_train.shape
print "test size = ", X_test.shape, y_test.shape
y_score = clf.fit(X_train, y_train).decision_function(X_test)
print clf
pred_class = clf.predict(X_test)
N_match = (pred_class == y_test).sum()
print "N_match = ", N_match
acc = 1. * N_match / len(pred_class)
print "Accuracy = ", acc

fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="best")
plt.show()

#ss = cross_validation.StratifiedShuffleSplit(classes, n_iter=10, test_size=1./3.)
#scores = cross_validation.cross_val_score(clf, data_svm, classes, cv=ss)
#print "Accuracy = ", scores.mean(), "+-", scores.std()