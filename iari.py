import numpy as np

from sklearn import tree
from sklearn.externals import joblib
from deap import benchmarks
from sklearn import preprocessing
import math
from sklearn import ensemble
import copy
from sklearn.preprocessing import Imputer
from sklearn.gaussian_process import GaussianProcess
from sklearn import cross_validation
from sklearn import datasets
from sklearn.datasets import fetch_mldata
import gc
import time
import matplotlib.pyplot as pl
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys
from sklearn.feature_selection import f_classif, f_regression
import urllib
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats



def IARI(X, y, regression=True,model_prefered="RandomForest",verbose=True):
	missingcolumnsize = len(X[0])
	missing = np.zeros(missingcolumnsize)
	training_x = X
	for j in range(missingcolumnsize): 	# For all features that something is missing, count the number of misses
		#missing = np.append(missing,0)
		#print missing
		for i in range(len(training_x)):
			if (math.isnan(training_x[i][j])):
				missing[j]+=1
	if verbose:
		print ("Missing data in each column: ",missing)

	#first impute using median or most frequent
	#
	start_time = time.time()
	

	# find the Univariate score of each feature using random forest feature importance
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	if (regression == False):
		imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
	imp.fit(training_x)
	imputed_x_1 = imp.transform(training_x)
	imputed_x_1_temp = copy.deepcopy(imputed_x_1)
	
	y = np.array(y).reshape(-1,1)
	#print(imputed_x_1_temp, y)
	extraforest,score = createClf(imputed_x_1_temp, y,imputed_x_1_temp,y,regression=regression)
	F = extraforest.feature_importances_
    
    
    
	for i in range(len(F)):
		if np.isnan(F[i]):
			F[i] = 0
	column_numbers = np.arange(missingcolumnsize)
	sorted_column_numbers = sorted(column_numbers, key=lambda best: -F[best-0]) 

	training_x_first = np.array(y)
	#print("y",y)
	for missing in sorted_column_numbers: 
		training_y_first = copy.deepcopy(training_x[:,missing])
		
		to_calc_i = []
		to_calc_x = []

		i = len(training_y_first)
		
		while i > 0:
			i -= 1
			if (math.isnan(training_y_first[i]) ):
				to_calc_x.append(training_x_first[i])
				to_calc_i.append(i)

		mask = np.ones(len(training_x_first), dtype=bool)
		mask[to_calc_i] = False
		training_x_first_mask=training_x_first[mask]

		mask = np.ones(len(training_y_first), dtype=bool)
		mask[to_calc_i] = False
		training_y_first = training_y_first[mask]#np.delete(training_y_first,(to_calc_i), axis=0)
		print(training_x_first,training_x_first_mask)
		clf, score = createClf(training_x_first_mask, training_y_first,training_x_first_mask,training_y_first,regression=regression)

		imputed = 0
		for i in to_calc_i:
			training_x[i,missing] = clf.predict(to_calc_x[imputed].reshape(1, -1))
			imputed += 1

		training_x_first = np.append(training_x_first, np.array([training_x[:,missing]]).T, axis=1) 

	if verbose:
		print("--- Imputation with Reduced Feature Models in %s seconds ---" % (time.time() - start_time))
	return training_x


#test function
def rastrigin_arg_vari(sol):
	#D1,D2,D3,D4,D5,D6,D7,D8,D9,D10 = sol[0], sol[1], sol[2], sol[3], sol[4], sol[5], sol[6], sol[7], sol[8], sol[9]
	Z = np.zeros(sol.shape[0])
	#print sol.shape[0]
	for i in xrange(sol.shape[0]):
		#print sol[i]
		Z[i] = benchmarks.rastrigin(sol[i])[0]
		#print Z[i]
	return Z

def createTraingSet(datasetname,seed):
	if len(datasetname.data) > 40000:
		datasetname.data = datasetname.data[:40000,:]
		datasetname.target = datasetname.target[:40000]

	std_scaler = StandardScaler()
	datasetname.data = std_scaler.fit_transform(datasetname.data,y=datasetname.target)

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(datasetname.data, datasetname.target, test_size=0.2, random_state=seed)

	return X_train, X_test, y_train, y_test


def createClf(training_x, training_y, test_x, test_y, printscore=False, regression=True,model_prefered="RandomForest"):
	#print "Initializing process"
	if (regression):
		#print "regression"
		if (model_prefered == "SVM"):
			clf = svm.SVR()
		elif (model_prefered=="Gaussian"):
			clf = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
		elif (model_prefered=="Gradient"):
			clf = GradientBoostingRegressor(n_estimators=100)
		else:
			clf = ensemble.RandomForestRegressor(n_estimators=100)
	else:
		#print "classifier"
		if (model_prefered == "SVM"):
			clf = svm.SVC()
		elif (model_prefered=="Gaussian"):
			exit() #cannot use gaussian for classification
			clf = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
		elif (model_prefered=="Gradient"):
			clf = GradientBoostingClassifier(n_estimators=100)
		else:
			clf = ensemble.RandomForestClassifier(n_estimators=100)
	clf.fit(training_x, training_y.ravel())
	#print "Done training"
	

	score = clf.score(test_x, test_y)
	if (printscore):
		print ("Score:", score)
	return clf, score


def clfPredict(clf,x):
	ant = clf.predict(x)
	return ant[0],1


from sklearn.neighbors import NearestNeighbors
def impute_NN2(trainingset, imputedmeanset):
	x = copy.deepcopy(trainingset)
	
	nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(imputedmeanset)
	#print x
	imputed = 0
	
	for i in range(len(x)):
		for j in range(len(x[i])):
			if (math.isnan(x[i][j])):
				distances, indices = nbrs.kneighbors(imputedmeanset[i])
				#count the number of missing values in the neighbours
				
				n1 = x[indices[0][1]]
				n2 = x[indices[0][2]]
				n1_m = float(len(n1))
				n2_m = float(len(n2))
				for n1_1 in n1:
					if math.isnan(n1_1):
						n1_m -= 1.0 
				for n2_1 in n2:
					if math.isnan(n2_1):
						n2_m -= 1.0 
				to = float(n1_m+n2_m)

				imputed += 1
				x[i][j] = imputedmeanset[indices[0][1]][j] * (n1_m / to) +  imputedmeanset[indices[0][2]][j] * (n2_m / to)

				
	#print "imputed", x, imputed
	return x


def impute_MODELS(trainingset, targetset, imputedmeanset, start_column_with_missing_data,end_column_with_missing_data):
	x = copy.deepcopy(trainingset)
	#print x.shape, targetset.shape

	x = np.hstack((x,np.array([targetset]).T)) #add the target to the features
	imputedmeanset = np.hstack((imputedmeanset,np.array([targetset]).T))

	modelarray = []
	modelinputs = []
	for j in range(start_column_with_missing_data,end_column_with_missing_data):
		model_training_set = np.hstack((imputedmeanset[:,:j],imputedmeanset[:,(j+1):]))
		modelinputs.append(model_training_set)
		model_training_target = imputedmeanset[:,j]
		clf,score = createClf(model_training_set, model_training_target,model_training_set,model_training_target)
		modelarray.append(clf)

	#imputation
	for i in range(len(x)):
		for j in range(start_column_with_missing_data,end_column_with_missing_data):
			if (math.isnan(x[i][j])):
				#model_input = np.hstack((imputedmeanset[i,:j],imputedmeanset[i,(j+1):]))
				x[i][j] = modelarray[j-start_column_with_missing_data].predict(modelinputs[j-start_column_with_missing_data][i])
	return x[:,:-1]



