from sklearn import metrics
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="Plot ROC curve from test data classes and classification scores.")
parser.add_argument('in_file_path', type=str, help='Path of the input npz file with data.')

args = parser.parse_args()

data = np.load(args.in_file_path)
#y_test, y_score_linear, y_score_rbf, y_score_sigmoid, y_score_poly = data['y_test'], data['y_score_linear'], data['y_score_rbf'], data['y_score_sigmoid'], data['y_score_poly']
y_test, y_score_linear, y_score_rbf, y_score_sigmoid = data['y_test'], data['y_score_linear'], data['y_score_rbf'], data['y_score_sigmoid']

fpr_linear, tpr_linear, _ = metrics.roc_curve(y_test, y_score_linear)
fpr_rbf, tpr_rbf, _ = metrics.roc_curve(y_test, y_score_rbf)
fpr_sigmoid, tpr_sigmoid, _ = metrics.roc_curve(y_test, y_score_sigmoid)
#fpr_poly, tpr_poly, _ = metrics.roc_curve(y_test, y_score_poly)

roc_auc_linear = metrics.auc(fpr_linear, tpr_linear)
roc_auc_rbf = metrics.auc(fpr_rbf, tpr_rbf)
roc_auc_sigmoid = metrics.auc(fpr_sigmoid, tpr_sigmoid)
#roc_auc_poly = metrics.auc(fpr_poly, tpr_poly)

plt.figure()
plt.plot(fpr_linear, tpr_linear, label='Linear SVC (area = %0.2f)' % roc_auc_linear)
plt.plot(fpr_rbf, tpr_rbf, label='RBF SVC (area = %0.2f)' % roc_auc_rbf)
plt.plot(fpr_sigmoid, tpr_sigmoid, label='Sigmoid SVC (area = %0.2f)' % roc_auc_sigmoid)
#plt.plot(fpr_poly, tpr_poly, label='Polynomial SVC (area = %0.2f)' % roc_auc_poly)
plt.plot([0,1], [0,1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="best")
plt.show()