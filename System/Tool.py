from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV # as GSCV
from sklearn.metrics import classification_report # as clfr
from sklearn.svm import SVC,NuSVC
from datetime import date
import sys
import os
import helper
import numpy as np
import pandas as pd
import math


'''
	>>> stock.Adj_Close,
	>>> stock.High,
	>>> stock.Low,
	>>> stock.EarningPerShare,
	>>> stock.Volatility5,
	>>> stock.Volatility10,
	>>> stock.Volatility25,
	>>> stock.dailyROR,
	>>> stock.HighROR,
	>>> stock.LowROR,
	>>> stock.alpha,
	>>> stock.beta,
	>>> stock.SharpeR,
	>>> stock.TreynorR,
	>>> stock.PVT

'''
USED_FEATURE = []

SVM_filename = "SVM_Classification.mdl"
DIVIDE_RATIO = 0.9



class DataProcessor():
	""" DataProcessor """
	def __init__(self, stock, window_size=10):
		self.window_size = window_size
		self.raw = self.filterFeature(stock=stock, used=USED_FEATURE)
		(self.feature, self.X_raw, self.y_raw, self.date_raw) = self.extractFeature(stock=stock, window_size=window_size)
		self.setIndexDate(stock=stock)

		self.X_train = self.X_raw[:int(len(self.X_raw)*DIVIDE_RATIO)]
		self.X_test = self.X_raw[int(len(self.X_raw)*DIVIDE_RATIO):]

		self.y_train = self.y_raw[:int(len(self.y_raw)*DIVIDE_RATIO)]
		self.y_test = self.y_raw[int(len(self.y_raw)*DIVIDE_RATIO):]



		# (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(self.X_raw, self.y_raw, test_size=0.3, random_state=0)
		# self.Model = SVC()
		# TODO

	def filterFeature(self, stock, used=USED_FEATURE):
		print USED_FEATURE
		# feature selection & date intercept
		print "[DataProcessor] feature selection & date intercept ..."
		raw = [stock.Adj_Close,
				stock.High,
				stock.Low,
				stock.EarningPerShare,
				stock.Volatility5,
				stock.Volatility10,
				stock.Volatility25,
				stock.dailyROR,
				stock.HighROR,
				stock.LowROR,
				stock.alpha,
				stock.beta,
				stock.SharpeR,
				stock.TreynorR,
				stock.PVT]
		assert(len(raw) == len(USED_FEATURE))
		tmp = []
		for i in range(len(USED_FEATURE)):
			if USED_FEATURE[i] == 1:
				tmp.append(raw[i])
		raw = tmp
		tmp = [len(raw[i]) for i in range(len(raw))]
		stock._end = min(tmp)-1
		del tmp
		return np.array(raw)

	def extractFeature(self, stock, window_size=10):
		# sample construction
		print "[DataProcessor] sample construction ..."
		x_feat_all_days = []
		for i in xrange(stock._end, stock._start-1, -1):
			day = stock.Date[i]
			x_feat_a_day = []
			for feat in self.raw:
				assert(feat.ndim == 1 or feat.ndim == 2)
				if feat.ndim == 1:
					x_feat_a_day.append(feat[i])
				elif feat.ndim == 2:
					use_day = feat[:, 0]
					index = np.argwhere(use_day==day)[0, 0]
					x_feat_a_day.append(float(feat[index, 1]))
			x_feat_all_days.insert(0, (day, x_feat_a_day))
		X_raw = []
		y_raw = []
		date_raw = []
		for i in xrange(stock._end-window_size, stock._start-1, -1):
			day = stock.Date[i]
			assert(day == x_feat_all_days[i-stock._start][0])
			x_sample = []
			for offset in range(window_size):
				x_sample.append(x_feat_all_days[i-stock._start+offset][1])
			x_sample = np.array(x_sample)
			X_raw.append(x_sample.reshape(x_sample.size))
			y_raw.append(int(stock.label[i, 1]))
			date_raw.append(day)

		return x_feat_all_days, np.array(X_raw), np.array(y_raw), np.array(date_raw)

	def setIndexDate(self, stock):
		# set Index_Start Date
		x = np.argwhere(self.date_raw==stock._index_date)
		try:
			assert(x.size == 1)
			self.predictNil = x[0, 0]
		except Exception, e:
			print "Fatel error: illegal trading day ..."
			raise e
		print "[DataProcessor] IndexDate", self.predictNil

	def getRawByCount(self, start_count, end_count):
		head = self.predictNil+start_count
		tail = self.predictNil+end_count
		try:
			assert(head <= tail and head >= 0 and tail < self.date_raw.size)
		except Exception, e:
			print "Fatal error: raw date split out of bound ..."
			raise e
		X_split = self.X_raw[head:tail+1]
		y_split = self.y_raw[head:tail+1]
		date_split = self.date_raw[head:tail+1]
		return X_split, y_split, date_split



	def training(self):
		pass


from Basic import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from FeatureSelection import *


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**i for i in range(-8,8)], 'C': [2**i for i in range(-8,8)]},
 					# {'kernel': ['linear'], 'C': [2**i for i in range(-8,9,2)]},
 					# {'kernel': ['poly'], 'gamma': [2**i for i in range(-8,9,2)], 'C': [2**i for i in range(-8,9,2)], 'degree':[2,3,4]},
 					]
classifiers = [
# ("Decision Tree",DecisionTreeClassifier(class_weight='balanced')), 
# ("Random Forest(entropy)",RandomForestClassifier(criterion = 'entropy', n_estimators=100,max_features = 'auto',n_jobs= 4,class_weight='balanced')),
# ("Extrenmely Forest(entropy)",ExtraTreesClassifier(criterion = 'entropy', n_estimators=100,max_features = 'auto',n_jobs= 4,class_weight='balanced')),

# ("Random Forest(gini)",RandomForestClassifier(criterion = 'gini', n_estimators=100,max_features = 'auto',n_jobs= 4,class_weight='balanced')),
# ("Random Forest",RandomForestClassifier(criterion = 'entropy', n_estimators=5000,max_features = 'auto',n_jobs= -1)),
# ("AdaBoost",AdaBoostClassifier( n_estimators=100,)),
# ("Gaussian Naive Bayes",GaussianNB()),
# ("LDA",LDA()),
# ("QDA",QDA()),
("GBDT",GradientBoostingClassifier(n_estimators=200, max_features = 'auto',)),
# ("SVM",GridSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=5)),
# ("SVM",NuSVC(class_weight='balanced')),

]


def offlineLearning_demo(interv =1):

	stk = Stock(600050, default_start_date, interv)
	dp = DataProcessor(stk, 7)
	print "dp.X_train length:" ,len(dp.X_train)

	# dp.X_train,dp.X_test = featureSelection (dp.X_train,dp.y_train,dp.X_test,dp.y_test,method = 'f_class',testmode = False,n_features_to_select = None)




	for name, clf in classifiers:
		clf.fit(dp.X_train, dp.y_train)
		# PredY = clf.predict(TestX)
		y_true, y_pred = dp.y_test, clf.predict(dp.X_test)


		print name
		if name == 'SVM':
			print clf.best_params_

		print classification_report(y_true, y_pred)
		accuracy = clf.score(dp.X_test, dp.y_test)
		print("\t\tAccuracy = %0.4f" % accuracy)

def onlineLearning_demo(onLine_batch_size = 1,interv =1):




	stk = Stock(600050, default_start_date, interv)
	dp = DataProcessor(stk, 7)

	# dp.X_raw,a = featureSelection (dp.X_raw,dp.y_raw,[],[],method = 'f_class',testmode = False,)



	train_batch_size = int(len(dp.X_raw)*DIVIDE_RATIO)
	train_batch_size = 100

	# const = 0.1 **(1.0/1000)
	sampleW = [1 for i in range(10)]+[0.5 for i in range(20)]+[0.2 for i in range(30)]+[0.1 for i in range(40)]
	print "train_batch_size: ", train_batch_size
	test_batch_size = onLine_batch_size

	for name,clf in classifiers:
		print name
		y_true = dp.y_test
		y_pred = []

		for step in range(int(len(dp.X_raw)*DIVIDE_RATIO),len(dp.X_raw),onLine_batch_size):
			sys.stdout.write('.'),
			sys.stdout.flush()
			trainHead = step-train_batch_size
			trainTail = step
			clf.fit(dp.X_raw[trainHead:trainTail],dp.y_raw[trainHead:trainTail],)
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
	USED_FEATURE_copy = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
	USED_FEATURE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	
	# forward_backward()
	# 
	for i in range(1,21):
		# offlineLearning_demo(interv = i)
		onlineLearning_demo(interv = i)





	


