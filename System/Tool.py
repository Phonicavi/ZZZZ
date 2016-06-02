from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV # as GSCV
from sklearn.metrics import classification_report # as clfr
from sklearn.svm import SVC
from datetime import date
import sys
import os
import helper
import numpy as np
import pandas as pd


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


class DataProcessor():
	""" DataProcessor """
	def __init__(self, stock, window_size=10, divide_ratio=default_divide_ratio):
		self.window_size = window_size
		self.divide_ratio = default_divide_ratio
		self.raw = self.filterFeature(stock=stock, used=USED_FEATURE)
		(self.feature, self.X_raw, self.y_raw, self.date_raw) = self.extractFeature(stock=stock, window_size=window_size)
		self.setIndexDate(stock=stock)
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
						print "Fatel error missing day ... "
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
			print "Fatel error: illegal trading day ..."
			raise e
		print "[DataProcessor] IndexDate", self.predictNil
		print "[DataProcessor] Predict: start from ", self.date_raw[self.predictNil]

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
		X_split = self.X_raw[head:tail+1]
		y_split = self.y_raw[head:tail+1]
		date_split = self.date_raw[head:tail+1]
		return X_split, y_split, date_split

	def getPrice(self, stock, date_count):
		return stock.Adj_Close[self.predictNil+date_count]


	def training(self, flag=False):
		(self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(self.X_raw, self.y_raw, test_size=0.3, random_state=0)
		if flag:
			self.Model = SVC(C=0.03125, gamma=3.0517578125e-05, kernel='rbf', probability=True, decision_function_shape='ovr')
			self.Model.fit(self.X_train, self.y_train)
		else:
			tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**i for i in range(-15,-4)], 'C': [2**i for i in range(-5,8)]}]
			self.Model = GridSearchCV(SVC(decision_function_shape='ovr'), tuned_parameters, cv=7)
			self.Model.fit(self.X_train, self.y_train)
			joblib.dump(self.Model, SVM_filename, compress = 3)
			print self.Model.decision_function(self.X_test)
			print self.Model.best_params_

		y_true, y_pred = self.y_test, self.Model.predict(self.X_test)
		print classification_report(y_true, y_pred)

		accuracy = self.Model.score(self.X_test, self.y_test)
		print("\t\tAccuracy = %0.4f" % accuracy)


if __name__ == '__main__':
	pass
