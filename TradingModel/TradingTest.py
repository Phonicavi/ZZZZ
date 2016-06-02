# -*- coding:utf-8 -*-  
import sys
sys.path,append('../System/')
from Basic import Stock
from copy import deepcopy

TRANSACTION_COST = .003

'''TODO
	dp.getPrice(stock, date_count=int)

	dp.predictNext(stock, pred_date_count=int)
'''

class Investor:
	def __init__(self,_name = 'ZZZZ', _intial_cash = 10000, _start_date = '2014-06-01', _stockcode = 600000, _interval = 1):
		self.name = _name
		self.now = 0
		self.interval = _interval
		self.stocks = Stock(SN = _stockcode, start_date = _start_date, interval = _interval)

		## 0 represent by prediction, 1 represent by real lbls
		self.ttlCash = [_intial_cash,_intial_cash]
		self.ttlShare = [0,0]
		self.PV = [self.ttlCash[0],self.ttlCash[0]]
		self.inital_PV = deepcopy(self.PV)
		

	def getPV(cash,share,price):
		return cash + share*price
		

	def LongOneShare(self,which):
		nowPrice = self.stocks.getPrice(stock=self.stocks, date_count=self.now)
		tax = nowPrice*TRANSACTION_COST
		self.ttlCash[which] -= (nowPrice+tax)
		self.ttlShare[which] += 1

	def SellAndShortOne(self,which):
		nowPrice = self.stocks.getPrice(stock=self.stocks, date_count=self.now)
		tax = nowPrice*(self.ttlShare[which]+1)*TRANSACTION_COST
		self.ttlCash[which] += ((self.ttlShare[which]+1)*nowPrice-tax)


		## should buy one back the next day
		tomoPrice[which] = self.stocks.getPrice(stock=self.stocks, date_count=self.now+1)
		tax = tomoPrice*TRANSACTION_COST
		self.ttlCash[which] -= (tomoPrice+tax)

		# set Share to zero
		self.ttlShare[which] = 0
		

	def TradeNext(self):
		today = self.now
		trendPred = predictNext(stock = self.stocks,pred_date_count = today)
		trendReal = int (self.stocks.getPrice(stock=self.stocks, date_count=self.now+1) > self.stocks.getPrice(stock=self.stocks, date_count=self.now))

		if trendPred:
			self.LongOneShare(which = 0)
		else:
			self.SellAndShortOne(which = 0)

		if trendReal:
			self.LongOneShare(which = 1)
		else:
			self.SellAndShortOne(which = 1)

		self.now = today + 1

	def getTotalROR(self):
		nowPrice = self.stocks.getPrice(stock=self.stocks, date_count=self.now)
		self.PV = [getPV(ttlCash[i],ttlShare[i],nowPrice) for i in range(2)]
		ttlROR = [ float(self.PV[i]-self.inital_PV[i])/self.inital_PV[i] for i in range(2)]

		return ttlROR

if __name__ == '__main__':
	pass







		