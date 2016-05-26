from scipy import stats
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from datetime import date
import sys
import os
import helper
import numpy as np
import pandas as pd


DATA_DIR = "../data/"

default_start_date = '2014-06-02'
'''
	>>> stock.Adj_Close,
	>>> stock.Volatility10,
	>>> stock.dailyROR,
	>>> stock.alpha,
	>>> stock.beta,
	>>> stock.SharpeR,
	>>> stock.TreynorR,
	>>> stock.PVT

'''
USED_FEATURE = [1, 1, 1, 1, 1, 1, 1, 1]

MP_filename = "MarketPortfolio.base"
SVM_filename = "SVM_Classification.mdl"



class MarketPortfolio:
	"""
		class for Stock:000001
		SHANGZHENG-ZHISHU
		class for Stock:000002
		SHANGZHENG-A
		class for Stock:000016
		SHANGZHENG-50

	"""
	def __init__(self):
		try:
			filename = DATA_DIR+"000001_sz.csv"
			self.raw = np.array(pd.read_csv(filename))
			(self._m, self._n)  = self.raw.shape
		except Exception, e:
			print Exception,":",e
		# expected return
		self.ROR = self.getROR()
		joblib.dump(self, MP_filename, compress = 3)

	def getROR(self, item=6, interval=1):
		"""item: 6-Adj Close"""
		return np.array([float(self.raw[i, item]-self.raw[i+interval, item])/self.raw[i+interval, item] for i in range(self._m-interval)])


class Stock:
	"""
		class for Stock

			SN : serial number
			_m : dimension_1 date
			_n : dimension_2 feature
			market : SHANGZHENG-ZHISHU
			_start : first available sample date
			_end : last available sample date

		basic_features : raw data

			Open, High, Low, Close, Volume, Adj_Close
			(note that Volume can be zero)

		superior_features

			Volatility : standard variance of the last t days
			(t = 5, 10, 25)
				Volatility5
				Volatility10
				Volatility25

			EarningPerShare

			dailyROR
			marketROR
			* HighROR
			* LowROR

			alpha
			beta
			SharpeR
			WilliamsR
			TreynorR
			PVT

		labels

			label : 1 or -1

	"""
	def __init__(self, SN, start_date, interval):
		SN_head = SN / 1000
		assert(SN_head == 600 or SN_head == 601 or SN_head == 603) # Shanghai Stock Exchange - A
		filename = DATA_DIR+str(SN)+"_ss.csv"
		assert(os.path.exists(filename))
		# basic
		try:
			self.raw = np.array(pd.read_csv(filename))
		except Exception, e:
			print Exception,":",e
		self.SN = int(SN)
		(self._m, self._n)  = self.raw.shape
		try:
			if not os.path.exists(MP_filename):
				mkt = MarketPortfolio()
			self.market = joblib.load(MP_filename)
		except Exception, e:
			print Exception,":",e
		self.marketROR = self.market.ROR
		# available data range
		self._start = interval
		self._end = self._m

		self.getFeatures()
		self.getLabel(interval=interval)
		self.dumpRaw(start_date=start_date)


	def getFeatures(self):
		self.Volatility5 = self.getVolatility(interval=5)
		self.Volatility10 = self.getVolatility(interval=10)
		self.Volatility25 = self.getVolatility(interval=25)
		self.EarningPerShare = self.getEarningPerShare()
		self.dailyROR = self.getROR()
		self.alpha, self.beta = self.getAlphaBeta(interval=100)
		self.HighROR = self.getROR(item=2)
		self.LowROR = self.getROR(item=3)
		self.SharpeR = self.getSharpeR()
		self.WilliamsR = self.getWilliamsR()
		self.TreynorR = self.getTreynorR()
		self.PVT = self.getPVT()

	def getLabel(self, item=6, interval=1):
		"""Formula: today_label = sign(future - today), item: 6-Adj Close"""
		label = [(1 if (self.raw[i-interval, item] > self.raw[i, item]) else 0) for i in xrange(interval, self._m)]
		for x in range(interval):
			label.insert(0, float('nan'))
		self.label = np.array(label)

	def dumpRaw(self, start_date=default_start_date):
		# date process
		self.Date = np.array(self.raw[:, 0])
		x = np.argwhere(self.Date==start_date)
		assert(x.size == 1)
		self.index = x[0, 0]
		# basic features
		self.Open = np.array(self.raw[:, 1])
		self.High = np.array(self.raw[:, 2])
		self.Low = np.array(self.raw[:, 3])
		self.Close = np.array(self.raw[:, 4])
		self.Volume = np.array(self.raw[:, 5])
		self.Adj_Close = np.array(self.raw[:, 6])
		# release raw & market
		self.raw = []
		self.market = []



	def getVolatility(self, item=6, interval=10):
		"""item: 6-Adj_Close, interval: 10days"""
		return np.array([self.raw[i:i+interval, item].std() for i in range(self._m-interval)])

	def getEarningPerShare(self, item=6, interval=1):
		"""item: 6-Adj_Close"""
		return np.array([float(self.raw[i, item]-self.raw[i+interval, item]) for i in range(self._m-interval)])

	def getROR(self, item=6, interval=1):
		"""item: 6-Adj_Close"""
		return np.array([float(self.raw[i, item]-self.raw[i+interval, item])/self.raw[i+interval, item] for i in range(self._m-interval)])

	def getAlphaBeta(self, interval=100):
		"""Formula: (cov(dailyROR, marketROR)/var(marketROR)) or linear-regression:intercept, slope"""
		linreg = np.array([stats.linregress(self.marketROR[i:i+interval], self.dailyROR[i:i+interval]) for i in range(min(len(self.dailyROR), len(self.marketROR))-interval)])
		return linreg[:, 0], linreg[:, 1]

	def getSharpeR(self, interval=10):
		"""Formula: (dailyROR - rfr)/volatility # risk_free_return = 0"""
		assert(interval == 5 or interval == 10 or interval == 25)
		volatility = self.Volatility10
		if interval == 5:
			volatility = self.Volatility5
		elif interval == 25:
			volatility = self.Volatility25
		return np.array([float(self.dailyROR[i])/volatility[i] for i in range(min(len(self.dailyROR), len(volatility)))])

	def getWilliamsR(self):
		"""Formula:((High - Close)/(High - Low)), item: 1-Open, 2-High, 3-Low, 4-Close"""
		return np.array([(float('nan') if (self.raw[i, 2]-self.raw[i, 3] == 0.0) else float(self.raw[i, 2]-self.raw[i, 4])*100/(self.raw[i, 2]-self.raw[i, 3])) for i in range(self._m)])

	def getTreynorR(self):
		"""Formula: (dailyROR - rfr)/beta # risk_free_return = 0"""
		return np.array([float(self.dailyROR[i])/self.beta[i] for i in range(min(len(self.dailyROR), len(self.beta)))])

	def getPVT(self):
		"""Accumulation of PV: ROR*Volume"""
		PV = np.array([(float(self.raw[i, 6]-self.raw[i+1, 6])/self.raw[i+1, 6])*self.raw[i, 5] for i in range(self._m-1)])
		for i in xrange(len(PV)-1, 0, -1):
			PV[i-1] += PV[i]
		return PV

	"""
		Set Values

	"""
	def setLabel(self, item=6, interval=1):
		"""Longer interval leads to fewer samples"""
		self.label = getLabel(item, interval)


	"""
		Train and Predict

	"""


class DataProcessor():
	""" DataProcessor """
	def __init__(self, stock, window_size=10):
		self.window_size = window_size
		self.raw = self.filterFeature(stock=stock, used=USED_FEATURE)
		(self.feature, self.X_raw, self.y_raw, self.date_raw) = self.extractFeature(stock=stock, window_size=window_size)
		(self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(self.X_raw, self.y_raw, test_size=0.3, random_state=0)
		self.Model = SVC()
		# TODO

	def filterFeature(self, stock, used=USED_FEATURE):
		# print " feature selection & date intercept ..."
		# feature selection & date intercept
		raw = [stock.Adj_Close,
				stock.Volatility10,
				stock.dailyROR,
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
		# print " sample construction ..."
		# sample construction
		x_feat_all_days = []
		for i in xrange(stock._end, stock._start-1, -1):
			x_feat_a_day = []
			for feat in self.raw:
				x_feat_a_day.append(feat[i])
			x_feat_all_days.append(x_feat_a_day)
		x_feat_all_days = np.array(x_feat_all_days)
		X_raw = []
		y_raw = []
		date_raw = []
		for i in xrange(stock._end-window_size, stock._start-1, -1):
			x_sample = []
			for offset in range(window_size):
				x_sample.append(x_feat_all_days[i-stock._start+1+offset])
			x_sample = np.array(x_sample)
			X_raw.append(x_sample.reshape(x_sample.size))
			y_raw.append(int(stock.label[i]))
			date_raw.append(stock.Date[i])

		return np.array(x_feat_all_days), np.array(X_raw), np.array(y_raw), np.array(date_raw)

	def training(self):
		pass




if __name__ == '__main__':
	stk = Stock(600050, default_start_date, 1)
	# print stk.SN
	# print stk.Volatility10
	# stk.getVolatility()

	dp = DataProcessor(stk, 10)

	model = SVC(probability=True, decision_function_shape='ovr', kernel='rbf', gamma=0.0078125, C=8)
	model.fit(dp.X_train, dp.y_train)

	y_true, y_pred = dp.y_test, model.predict(dp.X_test)
	print classification_report(y_true, y_pred)

	accuracy = model.score(dp.X_test, dp.y_test)
	print("\t\tAccuracy = %0.4f" % accuracy)

	'''
	# set param by cross-validation
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-2, 1e-1, 1], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	scores = ['precision', 'recall']

	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()

		# instaniation
		print('\t------')
		classifier = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='%s_weighted' % score)
		print('\t......')
		classifier.fit(dp.X_train, dp.y_train)
		print('\t------')

		print("Best parameters set found on development set:")
		print()
		print(classifier.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		for params, mean_score, scores in classifier.grid_scores_:
			print("%0.4f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
		print()

		# # check result
		accuracy = classifier.score(dp.X_test, dp.y_test)
		print("\t\tAccuracy = %0.4f" % accuracy)

		print("Detailed classification report:")
		print()
		print("Model is trained on the full development set.")
		print("Scores are computed on the full development set.")
		print()
		y_true, y_pred = dp.y_test, classifier.predict(dp.X_test)
		print(classification_report(y_true, y_pred))
		print()
	'''
