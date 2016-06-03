# -*- coding:utf-8 -*-  
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV # as GSCV
from sklearn.metrics import classification_report # as clfr
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib
from sklearn.svm import SVC,NuSVC
from datetime import date
import sys
import os
import helper
import numpy as np
import pandas as pd
from copy import deepcopy



'''
	>>> stock.Adj_Close,
	>>> stock.marketPrice,
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
USED_FEATURE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

SVM_filename = "SVM_Classification.mdl"
default_divide_ratio = 0.9

tuned_parameters = [
					{'kernel':['rbf'], 'gamma':[2**i for i in range(-8, 8)], 'C':[2**i for i in range(-8, 8)]},
 					# {'kernel':['linear'], 'C':[2**i for i in range(-8, 9, 2)]},
 					# {'kernel':['poly'], 'gamma':[2**i for i in range(-8, 9, 2)], 'C':[2**i for i in range(-8, 9, 2)], 'degree':[2, 3, 4]}
 					]
classifiers = [
				# ("Decision Tree", DecisionTreeClassifier(class_weight='balanced')), 
				# ("Random Forest(entropy)", RandomForestClassifier(criterion='entropy', n_estimators=100, max_features='auto', n_jobs=4, class_weight='balanced')),
				# ("Extrenmely Forest(entropy)", ExtraTreesClassifier(criterion='entropy', n_estimators=100, max_features='auto', n_jobs=4, class_weight='balanced')),

				# ("Random Forest(gini)", RandomForestClassifier(criterion='gini', n_estimators=100, max_features='auto', n_jobs=4, class_weight='balanced')),
				# ("Random Forest", RandomForestClassifier(criterion='entropy', n_estimators=5000, max_features='auto', n_jobs=-1)),
				("AdaBoost", AdaBoostClassifier(n_estimators=100)),
				# ("Gaussian Naive Bayes", GaussianNB()),
				# ("LDA", LDA()),
				# ("QDA", QDA()),
				# ("GBDT", GradientBoostingClassifier(n_estimators=200, max_features='auto')),
				# ("SVM", GridSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=5)),
				# ("SVM", NuSVC(class_weight='balanced'))
				]
clf = RandomForestClassifier(criterion='gini', n_estimators=100, max_features='auto', n_jobs=4, class_weight='balanced')



class DataProcessor():
	""" DataProcessor """
	def __init__(self, stock, window_size=10, divide_ratio=default_divide_ratio):
		self.window_size = window_size
		self.divide_ratio = default_divide_ratio
		self.raw = self.filterFeature(stock=stock, used=USED_FEATURE)
		(self.feature, self.X_raw, self.y_raw, self.date_raw) = self.extractFeature(stock=stock, window_size=window_size)
		self.setIndexDate(stock=stock)
		self.printInfo(stock=stock)
		self.splitRaw()
		# TODO

	def filterFeature(self, stock, used=USED_FEATURE):
		# feature selection & date intercept
		print "[DataProcessor] feature selection & date intercept ..."
		raw = [stock.Adj_Close,
				stock.marketPrice,
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
					x = np.argwhere(use_day==day)
					try:
						assert(x.size == 1)
						index = x[0, 0]
					except Exception, e:
						print "Fatal error: missing day ... "
						raise e
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
			print "Fatal error: illegal trading day ..."
			raise e

	def printInfo(self, stock):
		print "[DataProcessor] IndexDate", self.predictNil
		print "[DataProcessor] Predict: start from ", self.date_raw[self.predictNil]
		print "[DataProcessor] Predict interval: ", stock._interval, " Used window size: ", self.window_size

	def splitRaw(self):
		self.X_train = self.X_raw[:int(len(self.X_raw)*self.divide_ratio)]
		self.X_test = self.X_raw[int(len(self.X_raw)*self.divide_ratio):]
		self.y_train = self.y_raw[:int(len(self.y_raw)*self.divide_ratio)]
		self.y_test = self.y_raw[int(len(self.y_raw)*self.divide_ratio):]


	def getRawByCount(self, start_count, end_count):
		head = self.predictNil+start_count
		tail = self.predictNil+end_count
		try:
			assert(head <= tail and head >= 0 and tail < self.date_raw.size)
		except Exception, e:
			print "Fatal error: raw date split out of bound ..."
			raise e
		X_split = self.X_raw[head:tail]
		y_split = self.y_raw[head:tail]
		date_split = self.date_raw[head:tail]
		return X_split, y_split, date_split

	def getSingleRaw(self, date_count):
		pos = self.predictNil+date_count
		try:
			assert(pos >= 0 and pos < self.date_raw.size)
		except Exception, e:
			print "Fatal error: date position out of bound ..."
			raise e
		X_single = self.X_raw[pos]
		y_single = self.y_raw[pos]
		date_single = self.date_raw[pos]
		return X_single, y_single, date_single

	def getPriceByCount(self, stock, date_count):
		date_string = self.getDateStringByDateCount(date_count)
		x = np.argwhere(stock.Date==date_string)
		try:
			assert(x.size==1)
			index = x[0, 0]
		except Exception, e:
			print "Fatal error: date index out of bound ..."
			raise e
		return stock.Adj_Close[index]

	def getMaxDateCount(self):
		return self.date_raw.shape[0]-self.predictNil

	def predictNext(self, stock, pred_date_count, train_batch_size=100, use_NN=True):
		trainX, trainY, trainD = self.getRawByCount(pred_date_count-train_batch_size, pred_date_count);
		# print trainX[0]
		# print list(trainX)
		sc = StandardScaler()
		sc.fit(trainX)
		trainX = sc.transform(trainX)

		testX, testY, testD = self.getSingleRaw(pred_date_count)
		testX = testX.reshape(1, -1)
		testX = sc.transform(testX)

		if use_NN:
			from Power import NNet
			predY = NNet(TrainX=trainX, TrainY=trainY, TestX=testX)
		else:
			clf.fit(trainX, trainY)
			predY = clf.predict(testX)

		return predY[0], testY, testD


	def getDateCountByDateString(self, date_string):
		x = np.argwhere(self.date_raw==date_string)
		if x.size == 1:
			return x[0, 0]-self.predictNil
		else:
			index = 0
			for item in self.date_raw:
				if item > date_string:
					return index-self.predictNil
				index += 1
			return index-self.predictNil

	def getDateStringByDateCount(self, date_count):
		return self.date_raw[self.predictNil+date_count]



if __name__ == '__main__':
	pass
