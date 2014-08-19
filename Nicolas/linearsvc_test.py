from sklearn import svm, cross_validation
import os
import numpy as np
import itertools

"""
Small script for testing Linear SVC with cross-validation
"""

# Select file and load raw data into a matrix
file_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data_csv', 'GalaxyZooData.csv')
#data = np.genfromtxt(file_path, delimiter=',', names=True).transpose()
with open(file_path) as t_in:
	data = np.genfromtxt(itertools.islice(t_in, 500), delimiter=',', names=True).transpose()

# Generate a vector of classes
eliptical, spiral = data['p_el_debiased'] > .8, data['p_cs_debiased'] > .8
unknown = np.ones(len(eliptical), dtype=bool) & (~eliptical) & (~spiral)
classes = eliptical * 1 + spiral * 2 + unknown * 3

# Select training data from raw data without classification features... OR ELSE THAT WOULD BE CHEATING!! ;)
svn_names = [x for x in data.dtype.names if x != "p_el_debiased" and x != "p_cs_debiased"]
data_svm = data[svn_names]

clf = svm.LinearSVC()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_svm, classes)
print "training set = ", X_train.shape, y_train.shape
print "test size = ", X_test.shape, y_test.shape
clf.fit(data_svm, classes.transpose())
print clf