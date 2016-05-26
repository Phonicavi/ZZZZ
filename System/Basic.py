from scipy import stats
from sklearn.externals import joblib
import sys
import os
import helper
import numpy as np
import pandas as pd


DATA_DIR = "../data/"

class MarketPortfolio:
	"""
		class for Stock:000001
		SHANGZHENG-ZHISHU
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
		joblib.dump(self,'MarketPortfolio.mdl',compress = 3)

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

		basic_features : raw data

			Open
			High
			Low
			Close
			Volume
			Adj_Close

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

			label

	"""
	def __init__(self, SN):
		SN_head = SN / 1000
		# Shanghai Stock Exchange - A
		assert(SN_head == 600 or SN_head == 601 or SN_head == 603)

		filename = DATA_DIR+str(SN)+"_ss.csv"
		assert(os.path.exists(filename))

		# basic
		try:
			self.raw = np.array(pd.read_csv(filename))
		except Exception, e:
			print Exception,":",e
		self.SN = int(SN)
		(self._m, self._n)  = self.raw.shape
		self.market = joblib.load('MarketPortfolio.mdl')
		self.marketROR = self.market.ROR

		# features
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

		# labels
		self.label = self.getLabel()

		# raw
		self.dumpRaw()


	def dumpRaw(self):
		self.Open = np.array(self.raw[:, 1])
		self.High = np.array(self.raw[:, 2])
		self.Low = np.array(self.raw[:, 3])
		self.Close = np.array(self.raw[:, 4])
		self.Volume = np.array(self.raw[:, 5])
		self.Adj_Close = np.array(self.raw[:, 6])
		# release raw & market
		self.raw = []
		self.market = []


	def getLabel(self, item=6, interval=1):
		"""(today = future - today), item: 6-Adj Close"""
		label = [(1 if (self.raw[i-interval, item] > self.raw[i, item]) else (0 if (self.raw[i-interval, item] == self.raw[i, item]) else -1)) for i in xrange(interval, self._m)]
		label.insert(0, float('nan'))
		return np.array(label)


	def getVolatility(self, item=6, interval=10):
		"""item: 6-Adj Close, interval: 10days"""
		return np.array([self.raw[i:i+interval, item].std() for i in range(self._m-interval)])

	def getEarningPerShare(self, item=6, interval=1):
		"""item: 6-Adj Close"""
		return np.array([float(self.raw[i, item]-self.raw[i+interval, item]) for i in range(self._m-interval)])

	def getROR(self, item=6, interval=1):
		"""item: 6-Adj Close"""
		return np.array([float(self.raw[i, item]-self.raw[i+interval, item])/self.raw[i+interval, item] for i in range(self._m-interval)])

	def getAlphaBeta(self, interval=100):
		"""(cov(dailyROR, marketROR)/var(marketROR)) or linear-regression:intercept, slope"""
		linreg = np.array([stats.linregress(self.marketROR[i:i+interval], self.dailyROR[i:i+interval]) for i in range(min(len(self.dailyROR), len(self.marketROR))-interval)])
		return linreg[:, 0], linreg[:, 1]

	def getSharpeR(self, interval=10):
		"""(dailyROR - rfr)/volatility # risk_free_return = 0"""
		assert(interval == 5 or interval == 10 or interval == 25)
		volatility = self.Volatility10
		if interval == 5:
			volatility = self.Volatility5
		elif interval == 25:
			volatility = self.Volatility25
		return np.array([float(self.dailyROR[i])/volatility[i] for i in range(min(len(self.dailyROR), len(volatility)))])

	def getWilliamsR(self):
		"""1-Open, 2-High, 3-Low, 4-Close"""
		return np.array([(float('nan') if (self.raw[i, 2]-self.raw[i, 3] == 0.0) else float(self.raw[i, 2]-self.raw[i, 4])*100/(self.raw[i, 2]-self.raw[i, 3])) for i in range(self._m)])

	def getTreynorR(self):
		"""(dailyROR - rfr)/beta # risk_free_return = 0"""
		return np.array([float(self.dailyROR[i])/self.beta[i] for i in range(min(len(self.dailyROR), len(self.beta)))])

	def getPVT(self):
		"""accumulation of PV: ROR*Volume"""
		PV = np.array([(float(self.raw[i, 6]-self.raw[i+1, 6])/self.raw[i+1, 6])*self.raw[i, 5] for i in range(self._m-1)])
		for i in xrange(len(PV)-1, 0, -1):
			PV[i-1] += PV[i]
		return PV


if __name__ == '__main__':
	stk1 = Stock(600050)
	print stk1.SN
	print stk1.Volatility10
	stk1.getVolatility()

