'''
PCA-SVM pipeline - needs some more visualization, maybe
Reference: https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py
'''

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.svm import SVC
from util import load_data, model_checkpoint
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = load_data()
target_names = ["Incorrectly maksed", "Correctly masked"]

# PCA - keep the first 30 components
pca = PCA(n_components=30, svd_solver='full')
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Hyperparameters
print("Starting grid search...")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
clf = clf.fit(X_train, y_train)
print(f"Best estimator found by grid search: {clf.best_estimator_}")

# save the model
print("Saving the model to ./model folder...")
model_checkpoint(clf, 'pca')

y_predicted = clf.predict(X_test)

# print the classification report
print(classification_report(y_test, y_predicted, target_names=target_names))
# show the confusion matrix
plot_confusion_matrix(clf, X_test, y_test, display_labels=target_names,
            cmap=plt.cm.Blues, normalize='true')
plt.show()
