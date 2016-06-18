# -*- coding:utf-8 -*-  
import sys
sys.path.append('../System/')
from Basic import Stock
from copy import deepcopy
from Tool import DataProcessor, default_divide_ratio
from helper import sort_dict, load, save
from sklearn.metrics import classification_report, accuracy_score, f1_score
from progressbar import *
import random




TRANSACTION_COST = .003

TRUEY = []
PREDY = []

STOCK_POOL = [600570]
# STOCK_POOL = [600030, 600570, 600051, 600401, 600691, 600966, 600839]
# STOCK_POOL = [600210, 600487, 600598, 600419, 600572, 600718, 600756, 600536, 600776]



def getPV(cash,share,price):
	return cash + share*price

class Investor:
	def __init__(self,_name='ZZZZ', _initial_virtual_shares=1000, _start_date='2010-06-04', _end_date=None, _stockcode=600050, _interval=10):
		self.name = _name
		self.now = 0
		self.interval = _interval
		self.train_batch_size = 200

		'''
		self.stock_pool = []
		self.dp_pool = []
		self.maxcnt_pool = []
		for _code in STOCK_POOL:
			_stock = Stock(SN=_code, start_date=_start_date, interval=_interval)
			_dp = DataProcessor(stock=_stock, window_size=3)
			_maxcnt = _dp.getMaxDateCount()-self.interval-1
			self.stock_pool.append(_stock)
			self.dp_pool.append(_dp)
			self.maxcnt_pool.append(_maxcnt)

		'''
		self.stocks = Stock(SN=_stockcode, start_date=_start_date, interval=_interval)
		self.dp = DataProcessor(stock=self.stocks, window_size=3)
		# self.maxcnt = self.dp.getMaxDateCount()-self.interval-1
		if _end_date == None:
			self.maxcnt = self.dp.getMaxDateCount()-self.interval-1
		else:
			self.maxcnt = min(self.dp.getDateCountByDateString(_end_date), self.dp.getMaxDateCount()-self.interval-1)
		print "[Investor] Trainning: batch size =", self.train_batch_size



		## 0 represent by prediction, 1 represent by real lbls
		_initial_cash = self.dp.getPriceByCount(stock=self.stocks, date_count=0)*_initial_virtual_shares
		self.ttlCash = [_initial_cash, _initial_cash]
		self.ttlShare = [0, 0]
		self.PV = [self.ttlCash[0], self.ttlCash[0]]
		self.inital_PV = deepcopy(self.PV)






	def SellAndShort(self,which,shortshares = 1):
		nowPrice = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now)
		tax = nowPrice*(self.ttlShare[which]+shortshares)*TRANSACTION_COST
		self.ttlCash[which] += ((self.ttlShare[which]+shortshares)*nowPrice-tax)


		## should buy one back the next day
		tomoPrice = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now+self.interval)
		tax = tomoPrice*shortshares*TRANSACTION_COST
		self.ttlCash[which] -= (tomoPrice*shortshares+tax)

		# set Share to zero
		self.ttlShare[which] = 0

	def LongShares(self,which,nshares = 100):
		nowPrice = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now)
		tax = nowPrice*TRANSACTION_COST

		nshares = min(nshares,int(self.ttlCash[which]/(nowPrice+tax)))
		self.ttlCash[which] -= (nowPrice+tax)*nshares
		assert(self.ttlCash[which]>=0)
		self.ttlShare[which] += nshares

	def SellShares(self,which,nshares=sys.maxint):
		nshares = min(self.ttlShare[which],nshares)

		nowPrice = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now)
		tax = nowPrice*(nshares)*TRANSACTION_COST
		self.ttlCash[which] += ((nshares)*nowPrice-tax)
		# set Share to zero
		self.ttlShare[which] -= nshares
		

	def TradeNext(self, use_NN):
		today = self.now
		use_NN=False
		trendPredY, trendPred_prob, trendLabelY, nowD = self.dp.predictNext(stock=self.stocks, pred_date_count=today, train_batch_size=self.train_batch_size, use_NN=use_NN)
		trendReal = int(self.dp.getPriceByCount(stock=self.stocks, date_count=self.now+self.interval) >= self.dp.getPriceByCount(stock=self.stocks, date_count=self.now))

		TRUEY.append(trendReal)
		PREDY.append(trendPredY)

		# if (not trendReal==trendPredY and trendReal == False):
		# 	print trendPred_prob,'true=',trendReal,' pred=',trendPredY

		if trendPredY :
			self.LongShares(which=0)
		else:
			self.SellAndShort(which=0,shortshares=10)
			# self.SellShares(which=0)

		if trendReal:
			self.LongShares(which=1)
		else:
			self.SellAndShort(which=1,shortshares=10)
			# self.SellShares(which=1)

		self.now = today + self.interval
		# print self.now

	def getTotalROR(self):
		nowPrice = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now)
		self.PV = [getPV(self.ttlCash[i], self.ttlShare[i], nowPrice) for i in range(2)]
		ttlROR = [float(self.PV[i]-self.inital_PV[i])/self.inital_PV[i]*100 for i in range(2)]

		return ttlROR

def backtestHistory(_initial_virtual_shares, _start_date, _stockcode, _interval):
	ZZZZ = Investor(_name='ZZZZ', _initial_virtual_shares=_initial_virtual_shares, _start_date=_start_date, _stockcode=_stockcode, _interval=_interval)
	total = ZZZZ.maxcnt-ZZZZ.now
	pbar = ProgressBar(widgets=[' ', AnimatedMarker(), 'Predicting: ', Percentage()], maxval=total).start()
	while ZZZZ.now < ZZZZ.maxcnt:
	    pbar.update(ZZZZ.now)
	    time.sleep(0.01)
	    ZZZZ.TradeNext(use_NN=False)
	pbar.finish()

	print
	print classification_report(TRUEY, PREDY)
	f1 = f1_score(TRUEY, PREDY)
	accuracy = accuracy_score(TRUEY, PREDY)
	print "accuracy:", accuracy
	predROR = ZZZZ.getTotalROR()[0]
	realROR = ZZZZ.getTotalROR()[1]
	assert not (realROR == 0)
	print 'pred ROR:', predROR, '%', '\t|\treal ROR:', realROR, '%'

	return predROR, realROR, f1, accuracy


def StockSelection(stock_pool):
	predict_list = {}
	over_list = {}
	f1_list = {}
	accuracy_list = {}

	for _code in stock_pool:
		print "--------------------------------------------------------------------------------"
		(predR, realR, f1, accuracy) = backtestHistory(_initial_virtual_shares=100, _start_date='2014-06-04', _stockcode=_code, _interval=5)
		predict_list[_code] = predR
		over_list[_code] = float(predR/realR)
		f1_list[_code] = f1
		accuracy_list[_code] = accuracy
		print "--------------------------------------------------------------------------------"

	predict_list = sort_dict(predict_list)
	over_list = sort_dict(over_list)
	f1_list = sort_dict(f1_list)
	accuracy_list = sort_dict(accuracy_list)

	print "Predict Return rate: ", predict_list
	print "Over Return rate: ", over_list
	print "f1-score rank: ", f1_list
	print "accuracy rank: ", accuracy_list

	DIR_Storage = '../Results/'
	save(str(predict_list), DIR_Storage+"predict_list.list")
	save(str(over_list), DIR_Storage+"over_list.list")
	save(str(f1_list), DIR_Storage+"f1_list.list")
	save(str(accuracy_list), DIR_Storage+"accuracy_list.list")


if __name__ == '__main__':
	StockSelection(stock_pool=STOCK_POOL)






