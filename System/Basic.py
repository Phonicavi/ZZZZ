from scipy import stats
from sklearn.externals import joblib
from datetime import date
import sys
import os
import helper
import numpy as np
import pandas as pd


DATA_DIR = "../data/"

default_start_date = '2014-06-02'
default_base_type = 0
MARKET_INVENTORY = [("MarketPortfolio.base", DATA_DIR+"000001.csv"), 
					("MarketPortfolioA.base", DATA_DIR+"000002.csv"), 
					("MarketPortfolio50.base", DATA_DIR+"000016.csv")]

class MarketPortfolio:
	"""
		>>> 0: class for Stock:000001
				SHANGZHENG-ZHISHU
		>>> 1: class for Stock:000002
				SHANGZHENG-A
		>>> 2: class for Stock:000016
				SHANGZHENG-50

	"""
	def __init__(self, base_type=default_base_type):
		self.TYPE = base_type
		try:
			(MP_filename, filename) = MARKET_INVENTORY[self.TYPE]
			self.raw = np.array(pd.read_csv(filename))
		except Exception, e:
			print Exception,":",e
		# features
		joblib.dump(self, MP_filename, compress=3)
		# self.price = self.getPrice(item=6)
		# self.ROR = self.getROR()

	def getPrice(self, item=6):
		return np.array( [(self.raw[i, 0], self.raw[i, item]) for i in range(self.raw.shape[0])] )

	def getROR(self, interval=1):
		item = 3 # sina, item = 6; netease, item = 3
		return np.array([(self.raw[i, 0], float(self.raw[i, item]-self.raw[i+interval, item])/self.raw[i+interval, item]) for i in range(self.raw.shape[0]-interval)])


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
			marketPrice
			marketROR
			HighROR
			LowROR

			alpha
			beta
			SharpeR
			WilliamsR
			TreynorR
			PVT

		labels

			label : 1 or -1

	"""
	def __init__(self, SN, start_date, interval, base_type=default_base_type):
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
		print "[Stock] Serial Number:", self.SN
		try:
			if not os.path.exists(MARKET_INVENTORY[base_type][0]):
				mkt = MarketPortfolio(base_type)
			self.market = joblib.load(MARKET_INVENTORY[base_type][0])
			print "[Stock] market-portfolio loaded ..."
		except Exception, e:
			print Exception,":",e
		# check date matching
		try:
			assert(self.checkDate())
		except Exception, e:
			print "Fatal error dates not matched ... "
			raise e
		# available data range
		(self._m, self._n) = self.raw.shape
		self._start = interval
		self._end = self._m
		# dump raw_data
		self.dumpRaw(start_date=start_date)
		self.getFeatures()
		self.getLabel(interval=interval)
		# release raw & market
		self.raw = []
		self.market = []

	def checkDate(self):
		# clean
		print "[Stock] clean data ..."
		date_1 = self.market.raw[:, 0]
		date_2 = np.array(self.raw[:, 0])
		if not (date_1.shape == date_2.shape):
			print "[Stock] different length ... "
		new_market_raw = []
		new_raw = []
		i, j, k = 0, 0, 0
		while (i<date_1.shape[0]) and (j<date_2.shape[0]):
			if date_1[i] == date_2[j]:
				k += 1
				new_market_raw.append(self.market.raw[i])
				new_raw.append(self.raw[j])
			else:
				print "interrupt interval: ", k
				k = 0
				if date_1[i] > date_2[j]: # market has more
					print "market more - miss day: ", date_1[i]
					i += 1
				else: # self has more
					print "self more - miss day: ", date_2[j]
					j += 1
			i += 1
			j += 1
		self.market.raw = np.array(new_market_raw)
		self.raw = np.array(new_raw)
		del new_market_raw
		del new_raw
		return True

	def dumpRaw(self, start_date=default_start_date):
		# date process
		print "[Stock] date process ..."
		self.Date = np.array(self.raw[:, 0])
		x = np.argwhere(self.Date==start_date)
		try:
			assert(x.size == 1)
			self._index = x[0, 0]
		except Exception, e:
			print "Fatel error illegal trading day ... "
			raise e
		self._index_date = start_date
		# basic features
		self.Open = np.array(self.raw[:, 1])
		self.High = np.array(self.raw[:, 2])
		self.Low = np.array(self.raw[:, 3])
		self.Close = np.array(self.raw[:, 4])
		self.Volume = np.array(self.raw[:, 5])
		self.Adj_Close = np.array(self.raw[:, 6])


	def getFeatures(self):
		# calculate features
		print "[Stock] calculate features ..."
		self.marketPrice = self.market.getPrice(item=6)
		self.marketROR = self.market.getROR()
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
		# calculate labels
		print "[Stock] calculate label ..."
		label = [(self.raw[i, 0], (1 if (self.raw[i-interval, item] >= self.raw[i, item]) else 0)) for i in xrange(interval, self._m)]
		for x in range(interval):
			label.insert(0, (self.raw[x, 0], float('nan')))
		self.label = np.array(label)



	def getVolatility(self, item=6, interval=10):
		"""item: 6-Adj_Close, interval: 10days"""
		return np.array([(self.Date[i], self.raw[i:i+interval, item].std()) for i in range(self._m-interval)])

	def getEarningPerShare(self, item=6, interval=1):
		"""item: 6-Adj_Close"""
		return np.array([(self.Date[i], float(self.raw[i, item]-self.raw[i+interval, item])) for i in range(self._m-interval)])

	def getROR(self, item=6, interval=1):
		"""item: 6-Adj_Close"""
		return np.array([(self.Date[i], float(self.raw[i, item]-self.raw[i+interval, item])/self.raw[i+interval, item]) for i in range(self._m-interval)])

	def getAlphaBeta(self, interval=100):
		"""Formula: (cov(dailyROR, marketROR)/var(marketROR)) or linear-regression:intercept, slope"""
		linreg = np.array([stats.linregress(self.marketROR[i:i+interval][:, 1].astype(float), self.dailyROR[i:i+interval][:, 1].astype(float)) for i in range(min(len(self.dailyROR), len(self.marketROR))-interval)])
		Alpha = [(self.Date[i], linreg[i, 0]) for i in range(min(len(self.dailyROR), len(self.marketROR))-interval)]
		Beta = [(self.Date[i], linreg[i, 1]) for i in range(min(len(self.dailyROR), len(self.marketROR))-interval)]
		return np.array(Alpha), np.array(Beta)

	def getSharpeR(self, interval=10):
		"""Formula: (dailyROR - rfr)/volatility # risk_free_return = 0"""
		assert(interval == 5 or interval == 10 or interval == 25)
		volatility = self.Volatility10
		if interval == 5:
			volatility = self.Volatility5
		elif interval == 25:
			volatility = self.Volatility25
		return np.array([(self.Date[i], float(self.dailyROR[i, 1])/float(volatility[i, 1])) for i in range(min(len(self.dailyROR), len(volatility)))])

	def getWilliamsR(self):
		"""Formula:((High - Close)/(High - Low)), item: 1-Open, 2-High, 3-Low, 4-Close"""
		return np.array([(self.Date[i], (float('nan') if (self.raw[i, 2]-self.raw[i, 3] == 0.0) else float(self.raw[i, 2]-self.raw[i, 4])*100/(self.raw[i, 2]-self.raw[i, 3]))) for i in range(self._m)])

	def getTreynorR(self):
		"""Formula: (dailyROR - rfr)/beta # risk_free_return = 0"""
		return np.array([(self.Date[i], float(self.dailyROR[i, 1])/float(self.beta[i, 1])) for i in range(min(len(self.dailyROR), len(self.beta)))])

	def getPVT(self):
		"""Accumulation of PV: ROR*Volume"""
		PV = np.array([(float(self.raw[i, 6]-self.raw[i+1, 6])/self.raw[i+1, 6])*self.raw[i, 5] for i in range(self._m-1)])
		for i in xrange(len(PV)-1, 0, -1):
			PV[i-1] += PV[i]
		PVT = []
		for i in xrange(len(PV)-1, 0, -1):
			PVT.insert(0, (self.Date[i], PV[i]))
		return np.array(PVT)

	"""
		Set Values

	"""
	def setLabel(self, item=6, interval=1):
		"""Longer interval leads to fewer samples"""
		self.label = getLabel(item, interval)




if __name__ == '__main__':
	pass

