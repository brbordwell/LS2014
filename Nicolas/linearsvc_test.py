from sklearn import cross_validation
import os

"""
Small script for testing Linear SVC with cross-validation
"""

# Select file and load raw data into a matrix
file_path = os.path.join(os.path.dirname(__file__), 'data_csv', os.pardir, 'GalaxyZooData.csv')
data = np.genfromtxt(file_path, delimiter=',', names=True).transpose()

# Generate a vector of classes
eliptical, spiral = data['p_el_debiased'] > .8, data['p_cs_debiased'] > .8
unknown = np.ones(len(eliptical), dtype=bool) & (~eliptical) & (~spiral)
classes = data

# Select training data from raw data without classification features... OR ELSE THAT WOULD BE CHEATING!! ;)
svn_names = [x for x in data.dtype.names if x != "p_el_debiased" and x != "p_cs_debiased"]
data_svm = data[svn_names]

"clf = svm.LinearSVC()
X_train, X_test, y_train, y_text = cross_validation.train_test_split(data_svm, classes)
print "training set = ", X_train.shape, y_train.shape
print "test size = ", X_test.shape, y_test.shape
clf.fit(data_svm, classes)
print clf