from sklearn.externals import joblib
import sys
import os
import helper
import numpy as np
import pandas as pd


DATA_DIR = "../data/"

class MarketPortfolio:
	"""
		class for Stock:000016
		SHANGZHENG-50

	"""
	def __init__(self):
		try:
			filename = DATA_DIR+"000016_sz.csv"
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
	"""class for Stock"""
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

		# features
		self.Volatility5 = self.getVolatility(interval=5)
		self.Volatility10 = self.getVolatility(interval=10)
		self.Volatility25 = self.getVolatility(interval=25)

		self.EarningPerShare = self.getEarningPerShare()
		self.dailyROR = self.getROR()
		self.marketROR = self.market.ROR
		self.beta = self.getBeta()
		self.HighROR = self.getROR(item=2)
		self.LowROR = self.getROR(item=3)
		self.SharpeR = self.getSharpeR()
		self.WilliamsR = self.getWilliamsR()
		self.TreynorR = self.getTreynorR()
		# raw
		self.dumpRaw()


	def dumpRaw(self):
		self.Open = np.array(self.raw[:, 1])
		self.High = np.array(self.raw[:, 2])
		self.Low = np.array(self.raw[:, 3])
		self.Close = np.array(self.raw[:, 4])
		self.Volume = np.array(self.raw[:, 5])
		self.Adj_Close = np.array(self.raw[:, 6])
		# release raw
		self.raw = []
		# also remove market
		# self.market = []


	def getVolatility(self, item=6, interval=10):
		"""item: 6-Adj Close, interval: 10days"""
		return np.array([self.raw[i:i+interval, item].std() for i in range(self._m-interval)])

	def getEarningPerShare(self, item=6, interval=1):
		"""item: 6-Adj Close"""
		return np.array([float(self.raw[i, item]-self.raw[i+interval, item]) for i in range(self._m-interval)])

	def getROR(self, item=6, interval=1):
		"""item: 6-Adj Close"""
		return np.array([float(self.raw[i, item]-self.raw[i+interval, item])/self.raw[i+interval, item] for i in range(self._m-interval)])

	def getBeta(self):
		"""(dailyROR - rfr)/(marketROR - rfr) # risk_free_rate = 0"""
		return np.array([float(self.dailyROR[i]-self.marketROR[i]) for i in range(min(len(self.dailyROR), len(self.marketROR)))])

	def getSharpeR(self, interval=10):
		"""(dailyROR - rfr)/volatility # risk_free_rate = 0"""
		assert(interval == 5 or interval == 10 or interval == 25)
		volatility = self.Volatility10
		if interval == 5:
			volatility = self.Volatility5
		elif interval == 25:
			volatility = self.Volatility25
		return np.array([(sys.maxint if (volatility[i] == 0.0) else float(self.dailyROR[i])/volatility[i]) for i in range(min(len(self.dailyROR), len(volatility)))])

	def getWilliamsR(self):
		"""1-Open, 2-High, 3-Low, 4-Close"""
		return np.array([(sys.maxint if (self.raw[i, 2]-self.raw[i, 3] == 0.0) else float(self.raw[i, 2]-self.raw[i, 4])*100/self.raw[i, 2]-self.raw[i, 3]) for i in range(self._m)])

	def getTreynorR(self):
		"""(dailyROR - rfr)/beta # risk_free_rate = 0"""
		return np.array([(sys.maxint if (self.beta[i] == 0.0) else float(self.dailyROR[i])/self.beta[i]) for i in range(min(len(self.dailyROR), len(self.beta)))])



if __name__ == '__main__':
	stk1 = Stock(600050)
	print stk1.SN
	print stk1.Volatility10
	stk1.getVolatility()

