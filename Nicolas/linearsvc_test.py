from sklearn import svm
import os

file_path = os.path.join(os.path.dirname(__file__), 'data_csv', os.pardir, 'GalaxyZooData.csv')
data_svm = np.genfromtxt(file_path, delimiter=',', names=True).transpose()
clf = svm.LinearSVC()