import argparse
import itertools
import numpy as np
from sklearn import svm, cross_validation

"""
Applies SVC akgorithm to csv data, returns test data classes and classification scores
"""

parser = argparse.ArgumentParser(description="Apply SVC algorithm to csv data, returns test data classes and classification scores.")
parser.add_argument('csv_file_path', type=str, help='Path of the input csv file.')
parser.add_argument('out_file', type=str, help='Path of the output file.')
parser.add_argument('-n', dest='num_lines', default=5000, type=int, help='Maximum number of lines to read from csv file (default = 5000).')

args = parser.parse_args()

# Load data from csv to structured array
with open(args.csv_file_path) as t_in:
	data = np.genfromtxt(itertools.islice(t_in, args.num_lines+1), delimiter=',', names=True)

# From structured array to nparray
data_svm = data.view(np.float64).reshape(data.shape + (-1,))

# Identify elipticals and spirals
index_el, index_cs, index_objid = data.dtype.names.index('p_el_debiased'), data.dtype.names.index('p_cs_debiased'), data.dtype.names.index('objid')
elliptical, spiral = data_svm[:, index_el] > .8, data_svm[:, index_cs] > .8

# Generate classes
classes = np.zeros(len(spiral), dtype=bool) | spiral

# Remove classification features from data matrix... OR ELSE THAT WOULD BE CHEATING!! ;)
data_svm = np.delete(data_svm, 0, 1)
data_svm = np.delete(data_svm, 0, 1)
data_svm = np.delete(data_svm, 0, 1)

clf_linear = svm.SVC(kernel='linear')
clf_rbf = svm.SVC(kernel='rbf')
clf_sigmoid = svm.SVC(kernel='sigmoid')
clf_poly = svm.SVC(kernel='poly')

# Train with data doing cross validation
print "Splitting data..."
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_svm, classes, test_size=1./3.)
print "Training with linear kernel..."
y_score_linear = clf_linear.fit(X_train, y_train).decision_function(X_test)
print "Training with RBF kernel..."
y_score_rbf = clf_rbf.fit(X_train, y_train).decision_function(X_test)
print "Training with Sigmoid kernel..."
y_score_sigmoid = clf_sigmoid.fit(X_train, y_train).decision_function(X_test)
#print "Training with Polynomial kernel..."
#y_score_poly = clf_poly.fit(X_train, y_train).decision_function(X_test)

# Save output score vector
print "Saving data to %s..." % args.out_file
#np.savez_compressed(args.out_file, y_test=y_test, y_score_linear=y_score_linear, y_score_rbf=y_score_rbf, y_score_sigmoid=y_score_sigmoid, y_score_poly=y_score_poly)
np.savez_compressed(args.out_file, y_test=y_test, y_score_linear=y_score_linear, y_score_rbf=y_score_rbf, y_score_sigmoid=y_score_sigmoid)


#pred_class_linear = clf_linear.predict(X_test)
#N_match = (pred_class == y_test).sum()
#acc = 1. * N_match / len(pred_class)

#print "training set = ", X_train.shape, y_train.shape
#print "test size = ", X_test.shape, y_test.shape
#print clf
#print "N_match = ", N_match
#print "Accuracy = ", acc
