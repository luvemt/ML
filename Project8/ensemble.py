import numpy as np
import time
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import datapreprocesser

if len(sys.argv)!= 2:
	print ("Number of arguments should be 2!")
	sys.exit()

if sys.argv[1] == "digits" :
	X,y = datapreprocesser.prepare_digits()
else:
	X,y = datapreprocesser.prepare_data(sys.argv[1])
X_train, X_test, y_train, y_test = datapreprocesser.split_data(X,y)
X_train, X_test = datapreprocesser.standardize(X_train, X_test)

#Random forest
start_time = time.time()
forest = RandomForestClassifier(criterion = 'gini', n_estimators=100, random_state=1)
forest.fit(X_train, y_train)
total_time = time.time() - start_time
y_pred_forest = forest.predict(X_test)
accuracy = (y_test==y_pred_forest).sum()/len(y_test)
print("\nRANDOM FOREST:")
print("Accuracy: ", accuracy)
print("Running time: ", total_time)



#Bagging
tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)
bag = BaggingClassifier(base_estimator=tree,
						n_estimators=100,
						max_samples=1.0,
						max_features=1.0,
						bootstrap=True,
						bootstrap_features=False,
						n_jobs=1,
						random_state=1)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred= tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print("\nBAGGING:")
print("Decision tree train/test accuracies %.3f/%.3f" % (tree_train, tree_test))

start_time = time.time()
bag = bag.fit(X_train, y_train)
total_time = time.time() - start_time
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print("Bagging train/test accuracies %.3f/%.3f" % (bag_train, bag_test))
print("Running time: ",total_time)


#Adaboost
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=1)
ada = AdaBoostClassifier(base_estimator=tree,
						 n_estimators = 100,
						 learning_rate = 0.1,
						 random_state = 1)

start_time = time.time()
ada = ada.fit(X_train, y_train)
total_time = time.time() - start_time
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)

print("\nADABOOST")
print("Adaboost train/test accuracies %.3f/%.3f" % (ada_train, ada_test))
print("Running time: ",total_time)






