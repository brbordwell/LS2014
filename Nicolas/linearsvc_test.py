from sklearn import svm, cross_validation
import os
import numpy as np
import itertools

"""
Small script for testing Linear SVC with cross-validation
"""

NUM_OF_ROWS = 10000



# Select file and load raw data into a matrix
file_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data_csv', 'GalaxyZooData.csv')
#data = np.genfromtxt(file_path, delimiter=',', names=True).transpose()

#file_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data_csv', 'GalaxyZooData_2.csv')
with open(file_path) as t_in:
	data = np.genfromtxt(itertools.islice(t_in, NUM_OF_ROWS+1), delimiter=',', names=True).transpose()

# Generate a vector of classes
eliptical, spiral = data['p_el_debiased'] > .8, data['p_cs_debiased'] > .8
unknown = np.ones(len(eliptical), dtype=bool) & (~eliptical) & (~spiral)
classes = eliptical * 1 + spiral * 2 + unknown * 3

# Select training data from raw data without classification features... OR ELSE THAT WOULD BE CHEATING!! ;)
svn_names = [x for x in data.dtype.names if x != "p_el_debiased" and x != "p_cs_debiased" and x != "objid"]
data_svm = data[svn_names]

# From structured array to nparray
data_svm = data_svm.view(np.float64).reshape(data_svm.shape + (-1,))

clf = svm.LinearSVC()

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_svm, classes, test_size=1./3.)
#print "training set = ", X_train.shape, y_train.shape
#print "test size = ", X_test.shape, y_test.shape
#clf.fit(X_train, y_train)
#print clf
#pred_class = clf.predict(X_test)
#N_match = (pred_class == y_test).sum()
#print "N_match = ", N_match
#acc = 1. * N_match / len(pred_class)
#print "Accuracy = ", acc

ss = cross_validation.StratifiedShuffleSplit(classes, n_iter=10, test_size=1./3.)
scores = cross_validation.cross_val_score(clf, data_svm, classes, cv=ss)
print "Accuracy = ", scores.mean(), "+-", scores.std()