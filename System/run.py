# -*- coding:utf-8 -*-  
from Basic import Stock, MarketPortfolio
from Tool import DataProcessor, default_divide_ratio
import numpy as np
import pandas as pd
import os
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV # as GSCV
from sklearn.metrics import classification_report ,accuracy_score,f1_score # as clfr
from sklearn.svm import SVC,NuSVC
from datetime import date

from FeatureSelection import *



tuned_parameters = [
					{'kernel':['rbf'], 'gamma':[2**i for i in range(-8, 8)], 'C':[2**i for i in range(-8, 8)]},
 					# {'kernel':['linear'], 'C':[2**i for i in range(-8, 9, 2)]},
 					# {'kernel':['poly'], 'gamma':[2**i for i in range(-8, 9, 2)], 'C':[2**i for i in range(-8, 9, 2)], 'degree':[2, 3, 4]}
 					]
classifiers = [
				# ("Decision Tree", DecisionTreeClassifier(class_weight='balanced')), 
				# ("Random Forest(entropy)", RandomForestClassifier(criterion='entropy', n_estimators=100, max_features='auto', n_jobs=4, class_weight='balanced')),
				("Extrenmely Forest(entropy)", ExtraTreesClassifier(criterion='gini', n_estimators=150, max_features='auto', n_jobs=4, class_weight='balanced')),

				# ("Random Forest(gini)", RandomForestClassifier(criterion='gini', n_estimators=100, max_features='auto', n_jobs=4, class_weight='balanced')),
				# ("Random Forest", RandomForestClassifier(criterion='entropy', n_estimators=5000, max_features='auto', n_jobs=-1)),
				# ("AdaBoost", AdaBoostClassifier(n_estimators=100)),
				# ("Gaussian Naive Bayes", GaussianNB()),
				# ("LDA", LDA()),
				# ("QDA", QDA()),
				# ("GBDT", GradientBoostingClassifier(n_estimators=200, max_features='auto')),
				# ("SVM", GridSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=5)),
				# ("SVM", NuSVC(class_weight='balanced'))
				]

def play(self):
	cnt = 0
	error = 0
	val = []
	for i in xrange(600000,601000):
		filename = "../data/"+str(i)+"_ss.csv"
		if os.path.exists(filename):
			cnt += 1
			try:
				raw = np.array(pd.read_csv(filename))
			except Exception, e:
				error += 1
				# print Exception,":",e
				continue
			vol = np.array(raw[:, 5])
			zero = 0
			for x in range(len(vol)):
				if vol[x]==0:

					zero += 1
			val.append(zero)
			print "SN: ", i, "- have zero = ", zero


	print min(val)
	print max(val)

	print cnt
	print error

def offlineLearning_demo(SSN, interv =1,train_batch_size=100):

	stk = Stock(SN=SSN, start_date='2013-06-05', interval=interv,granularity=interv, granu_count=10)
	dp = DataProcessor(stock=stk, window_size=3)
	# print "dp.X_train length:" ,len(dp.X_train)
	trainX, trainY, trainD = dp.getRawByCount(0-train_batch_size, 0);
	print dp.getMaxDateCount()-interv-1
	testX, testY,D = dp.getRawByCount(0, (dp.getMaxDateCount()-interv-1));


	# dp.X_train,dp.X_test = featureSelection (dp.X_train,dp.y_train,dp.X_test,dp.y_test,method = 'f_class',testmode = False,n_features_to_select = None)

	for name, clf in classifiers:
		clf.fit(trainX, trainY)
		# PredY = clf.predict(TestX)
		y_true, y_pred = testY, clf.predict(testX)


		print name
		if name == 'SVM':
			print clf.best_params_

		print classification_report(y_true, y_pred)
		# print 'test error: ',accuracy_score(y_true,y_pred)

		print 'accuracy: ',accuracy_score(y_true,y_pred)
		print 'f1: ',f1_score(y_true,y_pred)

		# accuracy = clf.score(dp.X_test, dp.y_test)
		# print("\t\tAccuracy = %0.4f" % accuracy)

def onlineLearning_demo(SSN, onLine_batch_size=1, interv=1):
	stk = Stock(SN=SSN, start_date='2005-06-02', interval=interv, base_type=0)
	dp = DataProcessor(stock=stk, window_size=3)

	# train_batch_size = int(len(dp.X_raw)*default_divide_ratio)
	train_batch_size = 100
	# const = 0.1 **(1.0/1000)
	sampleW = [1 for i in range(10)]+[0.5 for i in range(20)]+[0.2 for i in range(30)]+[0.1 for i in range(40)]
	print "train_batch_size: ", train_batch_size
	test_batch_size = onLine_batch_size

	for name,clf in classifiers:
		print name
		y_true = dp.y_test
		y_pred = []

		for step in range(int(len(dp.X_raw)*default_divide_ratio),len(dp.X_raw),onLine_batch_size):
			sys.stdout.write('.'),
			sys.stdout.flush()
			trainHead = step-train_batch_size
			trainTail = step
			clf.fit(dp.X_raw[trainHead:trainTail].tolist(),dp.y_raw[trainHead:trainTail].tolist(),)
			# print clf.score(dp.X_raw[trainHead:trainTail],dp.y_raw[trainHead:trainTail])

			testHead = step
			testTail = step+test_batch_size
			tmpPred = list(clf.predict(dp.X_raw[testHead:testTail]))
			y_pred += tmpPred
			# print y_pred

		print 
		print classification_report(y_true, y_pred)
		# return float(classification_report(y_true,y_pred).split()[-2])

def forward_backward():
	global USED_FEATURE
	USED_FEATURE_copy = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	USED_FEATURE = USED_FEATURE_copy[:]
	best_fea = USED_FEATURE[:]
	best_f_score = onlineLearning_demo()
	print USED_FEATURE,best_f_score

	for idx in range(len(USED_FEATURE_copy)-1):
		USED_FEATURE = best_fea[:]
		USED_FEATURE[idx+1] = 1
		tmp_f_score = onlineLearning_demo()
		print USED_FEATURE,tmp_f_score

		if tmp_f_score>best_f_score:
			best_fea = USED_FEATURE[:]
			best_f_score = tmp_f_score
	print best_fea




if __name__ == '__main__':
	for size in [20,30,50,80,100,120,150,200,300]:
		offlineLearning_demo(600530,15,train_batch_size = size)
	# StockPool = [600030, 600100, 600570, 600051, 600401, 600691, 600966, 600839]

	# # stk = Stock(600050, '2005-06-02', 7)
	# # dp = DataProcessor(stk, 15)

	# # dp.training()

	
	# for stock in StockPool:
	# 	onlineLearning_demo(SSN=stock, onLine_batch_size=1, interv=20)
	
	# # offlineLearning_demo(interv = 20)


	# stk = Stock(SN=600839, start_date='2005-06-02', interval=1)
	# # print stk.dailyROR
	# # print stk.WilliamsR
	# # print stk.PVT
	# # print stk.trendCount
	# # print stk.contiTrend
	# # print stk.Date

	# '''
	# print stk._start, stk._end
	# print stk._index

	# print ' ---- '
	# print dp.X_raw.shape
	# print dp.date_raw

	

