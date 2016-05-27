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
USED_FEATURE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

SVM_filename = "SVM_Classification.mdl"



class DataProcessor():
	""" DataProcessor """
	def __init__(self, stock, window_size=10):
		self.window_size = window_size
		self.raw = self.filterFeature(stock=stock, used=USED_FEATURE)
		(self.feature, self.X_raw, self.y_raw, self.date_raw) = self.extractFeature(stock=stock, window_size=window_size)
		# (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(self.X_raw, self.y_raw, test_size=0.3, random_state=0)
		# self.Model = SVC()
		# TODO

	def filterFeature(self, stock, used=USED_FEATURE):
		# feature selection & date intercept
		print "[DataProcessor] feature selection & date intercept ..."
		raw = [stock.Adj_Close,
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
			# 
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
		# return x_feat_all_days
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


	def training(self):
		pass


if __name__ == '__main__':
	pass

