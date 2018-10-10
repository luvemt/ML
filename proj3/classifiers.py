from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import datapreprocess
import time
import sys




if len(sys.argv)!= 2:
	print ("Number of arguments should be 2")
	sys.exit()


X,y = datapreprocess.prepare_data(sys.argv[1])
X_train, X_test, y_train, y_test = datapreprocess.split_data(X,y)
X_train_std, X_test_std = datapreprocess.standardize(X_train, X_test)

 	

#Perceptron
pc = Perceptron(tol = 1e-3, random_state=0, max_iter=10000)
start_time = time.time()
pc.fit(X_train_std, y_train)
total_time = time.time() - start_time
y_pred_pc = pc.predict(X_test_std)
accuracy = (y_test == y_pred_pc).sum()/len(y_test)
print("\nPercetron:")
print("Accuracy:", accuracy)
print("Runtime:", total_time)

#Linear SVM
svm = LinearSVC( random_state=1, C=1.0, max_iter = 10000)
start_time = time.time()
svm.fit(X_train_std, y_train)
total_time = time.time() - start_time
y_pred_lsvm = svm.predict(X_test_std)
accuracy = (y_test == y_pred_lsvm).sum()/len(y_test)
print("\nLinear SVM:")
print("Accuracy:", accuracy)
print("Runtime:", total_time)


#NonLinear SVM
svm = SVC(kernel='rbf', gamma = 0.1, random_state=1, C=1.0, max_iter=10000)
start_time = time.time()
svm.fit(X_train_std, y_train)
total_time = time.time() - start_time
y_pred_svm = svm.predict(X_test_std)
accuracy = (y_test == y_pred_svm).sum()/len(y_test)
print("\nNonLinear SVM:")
print("Accuracy:", accuracy)
print("Runtime:", total_time)

#K nearest neighbor
knn = KNeighborsClassifier(n_neighbors = 5, p=2, metric='minkowski')

knn.fit(X_train_std, y_train)
start_time = time.time()
y_pred_knn = knn.predict(X_test_std)
total_time = time.time() - start_time
accuracy = (y_test == y_pred_knn).sum()/len(y_test)
print("\nKNNeighbors:")
print("Accuracy:", accuracy)
print("Runtime:", total_time)


#logistic regression
lr = LogisticRegression(C=100.0, solver = 'lbfgs', multi_class='multinomial',random_state=1, max_iter=1000)
start_time = time.time()
lr.fit(X_train_std, y_train)
total_time = time.time() - start_time
y_pred_lr = lr.predict(X_test_std)
accuracy = (y_test == y_pred_lr).sum()/len(y_test)
print("\nLogisticRegression:")
print("Accuracy:", accuracy)
print("Runtime:", total_time)


#decision tree
tree = DecisionTreeClassifier(criterion='gini', max_depth=100, random_state=1)
start_time = time.time()
tree.fit(X_train, y_train)
total_time = time.time() - start_time
y_pred_tree = tree.predict(X_test)
accuracy = (y_test == y_pred_tree).sum()/len(y_test)
print("\nDecision Tree:")
print("Accuracy:", accuracy)
print("Runtime:", total_time)



