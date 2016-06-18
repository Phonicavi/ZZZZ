# -*- coding:utf-8 -*-  
from scipy import stats
from sklearn.externals import joblib
from datetime import date
import sys
import os
import helper
import numpy as np
import pandas as pd
from copy import deepcopy


DATA_DIR = "../data/"

default_start_date = '2014-06-02'
default_base_type = 0
MARKET_INVENTORY = [("../System/MarketPortfolio.base", DATA_DIR+"000001.csv"), 
					("../System/MarketPortfolioA.base", DATA_DIR+"000002.csv"), 
					("../System/MarketPortfolio50.base", DATA_DIR+"000016.csv")]

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

	def getPrice(self, item=6):
		return np.array([(self.raw[i, 0], self.raw[i, item]) for i in range(self.raw.shape[0])] )

	def getROR(self, interval=1):
		item = 3 # yahoo, item = 6; netease, item = 3
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

			alpha, beta
			SharpeR
			WilliamsR
			TreynorR
			PVT
			trendCount
			contiTrend

		labels

			label : 1 or 0

	"""
	def __init__(self, SN, start_date=default_start_date, interval=5, base_type=default_base_type, granularity=1, granu_count=10):
		self.resource = "netease" # "yahoo"
		SN_head = SN / 1000
		assert(SN_head == 600 or SN_head == 601 or SN_head == 603) # Shanghai Stock Exchange - A
		if self.resource == "yahoo":
			filename = DATA_DIR+str(SN)+"_ss.csv"
		else:
			DATA_DIR = "../data163/"
			filename = DATA_DIR+"0"+str(SN)+".csv"
		assert(os.path.exists(filename))
		# basic
		try:
			self.raw = np.array(pd.read_csv(filename))
		except Exception, e:
			print Exception, ":", e
		self.SN = int(SN)
		try:
			if not os.path.exists(MARKET_INVENTORY[base_type][0]):
				mkt = MarketPortfolio(base_type)
			self.market = joblib.load(MARKET_INVENTORY[base_type][0])
#			print "[Stock] market-portfolio loaded ..."
		except Exception, e:
			print Exception,":",e
		# check date matching
		# self.cleanDate()
		# print self.market
		try:
			assert(self.cleanDate())
		except Exception, e:
			# print e
			print "Fatal error: dates not matched ... "
			raise e
		# available data range
		(self._m, self._n) = self.raw.shape
		self._interval = interval
		self._start = self._interval
		self._end = self._m
		self._granular = granularity
		self._granu_count = granu_count
		# dump raw_data
		self.dumpRaw(start_date=start_date)
		self.getFeatures()
		self.getLabel(interval=interval)
		# release raw & market
		self.raw = []
		self.market = []
		# print info
		print "[Stock] Serial Number:", self.SN
		print "[Stock] Name:", self.Name

	def cleanDate(self):
		# clean
#		print "[Stock] clean data ..."
		date_1 = self.market.raw[:, 0]
		date_2 = np.array(self.raw[:, 0])
		if not (date_1.shape == date_2.shape):
#			print "[Stock] different length ... "
			pass
		new_market_raw = []
		new_raw = []
		i, j, k = 0, 0, 0
		while (i<date_1.shape[0]) and (j<date_2.shape[0]):
			if date_1[i] == date_2[j]:
				# k += 1
				if self.raw[j, 10] == 0:
					# print "stop trading - no data: ", date_2[j]
					i += 1
					j += 1
					continue
				new_market_raw.append(self.market.raw[i])
				new_raw.append(self.raw[j])
			else:
				# print "interrupt interval: ", k
				# k = 0
				if date_1[i] > date_2[j]:
					# print "market more - miss day: ", date_1[i]
					i += 1
					continue
				else:
					# print "self more - miss day: ", date_2[j]
					j += 1
					continue
			i += 1
			j += 1
		self.market.raw = np.array(new_market_raw)
		self.raw = np.array(new_raw)
		del new_market_raw
		del new_raw
		return True

	def dumpRaw(self, start_date):
		# date process
#		print "[Stock] date process ..."
		self.Date = np.array(self.raw[:, 0])
		x = np.argwhere(self.Date==start_date)
		if x.size == 1:
			self._index = x[0, 0]
			self._index_date = start_date
		else:
			_index = 0
			for item in self.Date:
				if item < start_date:
					self._index = _index
					break
				_index += 1
			self._index = _index
			self._index_date = self.Date[_index]
			print "[*Stock] start_date forward modified ... "
		# basic features
		if self.resource == "yahoo":
			self.Open = np.array(self.raw[:, 1])
			self.High = np.array(self.raw[:, 2])
			self.Low = np.array(self.raw[:, 3])
			self.Close = np.array(self.raw[:, 4])
			self.Volume = np.array(self.raw[:, 5])
			self.Adj_Close = np.array(self.raw[:, 6])
		else:
			self.Name = self.raw[0, 2].decode('cp936')
			self.Open = np.array(self.raw[:, 6])
			self.High = np.array(self.raw[:, 4])
			self.Low = np.array(self.raw[:, 5])
			self.Close = np.array(self.raw[:, 3])
			self.Volume = np.array(self.raw[:, 10])
			self.Adj_Close = self.Close


	def getFeatures(self):
		# calculate features
#		print "[Stock] calculate features ..."
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
		self.trendCount = self.getTrendCount(granular=self._granular, granu_count=self._granu_count)
		self.contiTrend = self.getContinualTrend(granular=self._granular)

	def getLabel(self, interval=1):
		"""Formula: today_label = sign(future - today), Adj_Close"""
		# calculate labels
#		print "[Stock] calculate label ..."
		label = [(self.raw[i, 0], (1 if (self.Adj_Close[i-interval] >= self.Adj_Close[i]) else 0)) for i in xrange(interval, self._m)]
		for x in range(interval):
			label.insert(0, (self.raw[x, 0], float('nan')))
		self.label = np.array(label)


	def getVolatility(self, interval=10):
		"""Adj_Close, interval: 10days"""
		return np.array([(self.Date[i], self.Adj_Close[i:i+interval].std()) for i in range(self._m-interval)])

	def getEarningPerShare(self, interval=1):
		"""Adj_Close difference"""
		return np.array([(self.Date[i], float(self.Adj_Close[i]-self.Adj_Close[i+interval])) for i in range(self._m-interval)])

	def getROR(self, item=6, interval=1):
		""" * """
		if self.resource != "yahoo":
			x = {1:6, 2:4, 3:5, 4:3, 5:10, 6:3}
			item = x[item]
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
		""" * """
		(x, y, z) = (4, 5, 3)
		if self.resource == "yahoo":
			(x, y, z) = (2, 3, 4)
		return np.array([(self.Date[i], (float('nan') if (self.raw[i, x]-self.raw[i, y] == 0.0) else float(self.raw[i, x]-self.raw[i, z])*100/(self.raw[i, x]-self.raw[i, y]))) for i in range(self._m)])

	def getTreynorR(self):
		"""Formula: (dailyROR - rfr)/beta # risk_free_return = 0"""
		return np.array([(self.Date[i], float(self.dailyROR[i, 1])/float(self.beta[i, 1])) for i in range(min(len(self.dailyROR), len(self.beta)))])

	def getPVT(self):
		"""Accumulation of PV: ROR*Volume"""
		(x, y) = (3, 10)
		if self.resource == "yahoo":
			(x, y) = (6, 5)
		PV = np.array([(float(self.raw[i, x]-self.raw[i+1, x])/self.raw[i+1, x])*self.raw[i, y] for i in range(self._m-1)])
		for i in xrange(len(PV)-1, 0, -1):
			PV[i-1] += PV[i]
		PVT = []
		for i in xrange(len(PV)-1, 0, -1):
			PVT.insert(0, (self.Date[i], PV[i]))
		return np.array(PVT)

	def getTrendCount(self, granular, granu_count):
		""" * """
		TCount = []
		for i in range(len(self.Adj_Close)-granular*granu_count):
			time = i
			tc = 0
			for x in range(granu_count):
				time += granular
				tc += (1 if (self.Adj_Close[time-granular] >= self.Adj_Close[time]) else -1)
			# print (self.Date[i], tc)
			TCount.append((self.Date[i], tc))
		return np.array(TCount)

	def	getContinualTrend(self, granular):
		""" * """
		ContiTrend = []
		for i in range(len(self.Adj_Close)-granular):
			time = i
			trend = (1 if (self.Adj_Close[time] >= self.Adj_Close[time+granular]) else -1)
			ct = 0
			while time+granular < len(self.Adj_Close):
				sign = (1 if (self.Adj_Close[time] >= self.Adj_Close[time+granular]) else -1)
				if sign != trend:
					break
				ct += sign
				time += granular
			# print (self.Date[i], ct)
			ContiTrend.append((self.Date[i], ct))
		return np.array(ContiTrend)

	"""
		Set Values

	"""
	def setLabel(self, item=6, interval=1):
		"""Longer interval leads to fewer samples"""
		self.label = getLabel(item, interval)




if __name__ == '__main__':
	pass

