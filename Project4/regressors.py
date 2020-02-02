from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import datapreprocess
import numpy as np
import time
import sys



if len(sys.argv)!= 2:
	print ("Number of arguments should be 2")
	sys.exit()

if sys.argv[1] == "housing.data.txt" :
	X,y = datapreprocess.prepare_data(sys.argv[1])
else:
	X,y = datapreprocess.prepare_cal_data(sys.argv[1])
X_train, X_test, y_train, y_test = datapreprocess.split_data(X,y)
#X_train_std, X_test_std = datapreprocess.standardize(X_train, X_test)

sc_x = StandardScaler()
sc_y = StandardScaler()
y_tr_2d=y_train [ : , np . newaxis ]   
y_test_2d=y_test [ : , np . newaxis ]   
sc_x.fit(X_train)
sc_x.fit(X_test)
sc_y.fit(y_tr_2d)
sc_y.fit(y_test_2d)

X_tr_std = sc_x.transform(X_train)
X_test_std = sc_x.transform(X_test)
y_tr_std = sc_y.transform(y_tr_2d).flatten()
y_test_std = sc_y.transform(y_test_2d).flatten()

#For printing results
def print_results(rgrs, r2_tr,r2_test, mse_tr, mse_test, total_time):
	print("\n", rgrs)
	print("R2_train: ", r2_tr)
	print("R2_test: ", r2_test)
	print("MSE_train: ", mse_tr)
	print("MS_test: ", mse_test)
	print("Runtime: ", total_time)


#Linear Regression
lr_model = LinearRegression()
start_time = time.time()
lr_model.fit(X_tr_std, y_tr_std)
total_time = time.time() - start_time
r2_tr = lr_model.score(X_tr_std, y_tr_std)
r2_test = lr_model.score(X_test_std, y_test_std)
y_tr_predict = lr_model.predict(X_tr_std)
y_test_predict = lr_model.predict(X_test_std)
mse_tr = mean_squared_error(y_tr_predict, y_tr_std)
mse_test = mean_squared_error(y_test_predict, y_test_std)
print_results("Linear Regressor: ", r2_tr, r2_test, mse_tr,mse_test, total_time)

 #RansacRegregressor
ransac  = RANSACRegressor(LinearRegression(), max_trials=100, min_samples =50, loss='absolute_loss', residual_threshold=5.0, random_state=1)
start_time = time.time()
ransac.fit(X_tr_std, y_tr_std)
total_time = time.time() - start_time
r2_tr = ransac.score(X_tr_std, y_tr_std)
r2_test = ransac.score(X_test_std, y_test_std)
y_tr_predict = ransac.predict(X_tr_std)
y_test_predict = ransac.predict(X_test_std)
mse_tr = mean_squared_error(y_tr_predict, y_tr_std)
mse_test = mean_squared_error(y_test_predict, y_test_std)
print_results("Ransac Regressor: ", r2_tr, r2_test, mse_tr,mse_test, total_time)

#Ridge 
ridge = Ridge(alpha = 1.0)
start_time = time.time()
ridge.fit(X_tr_std, y_tr_std)
total_time = time.time() - start_time
r2_tr = ridge.score(X_tr_std, y_tr_std)
r2_test = ridge.score(X_test_std, y_test_std)
y_tr_predict = ridge.predict(X_tr_std)
y_test_predict = ridge.predict(X_test_std)
mse_tr = mean_squared_error(y_tr_predict, y_tr_std)
mse_test = mean_squared_error(y_test_predict, y_test_std)
print_results("Ridge Regressor: ", r2_tr, r2_test, mse_tr,mse_test, total_time)

#Lasso
lasso = Lasso(alpha = 2.0)
start_time = time.time()
lasso.fit(X_tr_std, y_tr_std)
total_time = time.time() - start_time
r2_tr = lasso.score(X_tr_std, y_tr_std)
r2_test = lasso.score(X_test_std, y_test_std)
y_tr_predict = lasso.predict(X_tr_std)
y_test_predict = lasso.predict(X_test_std)
mse_tr = mean_squared_error(y_tr_predict, y_tr_std)
mse_test = mean_squared_error(y_test_predict, y_test_std)
print_results("Lasso Regressor: ", r2_tr, r2_test, mse_tr,mse_test, total_time)

#Non-Linear 
quartic = PolynomialFeatures(degree=3)
lr_quartic = LinearRegression()
start_time = time.time()
X_quartic_tr = quartic.fit_transform(X_tr_std)
X_quartic_test = quartic.fit_transform(X_test_std)
lr_quartic.fit(X_quartic_tr, y_tr_std)
total_time = time.time() - start_time
r2_tr = lr_quartic.score(X_quartic_tr,y_tr_std)
r2_test = lr_quartic.score(X_quartic_test,y_test_std)
y_tr_predict = lr_quartic.predict(X_quartic_tr)
y_test_predict = lr_quartic.predict(X_quartic_test)
mse_tr = mean_squared_error(y_tr_predict, y_tr_std)
mse_test = mean_squared_error(y_test_predict, y_test_std)
print_results("Non-Linear Regressor: ", r2_tr, r2_test, mse_tr,mse_test, total_time)


#Normal Equation solution
if sys.argv[1] == "housing.data.txt":
	onevec = np.ones((X_tr_std.shape[0]))
	onevec = onevec[:,np.newaxis]
	Xb = np.hstack((onevec, X_tr_std))

	w = np.zeros(X_tr_std.shape[1])
	start_time = time.time()
	z = np.linalg.inv(np.dot(Xb.T, Xb))
	w = np.dot(z, np.dot(Xb.T,y_tr_std))
	total_time = time.time() - start_time


	y_predict = (np.dot(X_tr_std,w[1:]) + w[0])
		
	mse = mean_squared_error(y_predict, y_tr_std)
	r2 = r2_score(y_predict, y_tr_std)
	print("\n Normal Equation:")
	print("R2: ", r2)
	print("MSE: ", mse)
	print("Runtime: ", total_time)











