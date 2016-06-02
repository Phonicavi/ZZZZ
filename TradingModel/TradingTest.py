# -*- coding:utf-8 -*-  
import sys
sys.path.append('../System/')
from Basic import Stock
from copy import deepcopy
from Tool import DataProcessor, default_divide_ratio
from sklearn.metrics import classification_report,accuracy_score



TRANSACTION_COST = .003

'''TODO
	dp.getPriceByCount(stock, date_count=int)

	dp.predictNext(stock, pred_date_count=int)
'''

TRUEY = []
PREDY = []


def getPV(cash,share,price):
	return cash + share*price

class Investor:
	def __init__(self,_name='ZZZZ', _initial_virtual_shares=1000, _start_date='2010-06-04', _stockcode=600050, _interval=10):
		self.name = _name
		self.now = 0
		self.interval = _interval
		self.stocks = Stock(SN=_stockcode, start_date=_start_date, interval=_interval)

		self.dp = DataProcessor(stock=self.stocks, window_size=3)

		self.maxcnt = self.dp.getMaxDateCount()-self.interval-1


		## 0 represent by prediction, 1 represent by real lbls
		_initial_cash = self.dp.getPriceByCount(stock=self.stocks, date_count=0)*_initial_virtual_shares
		self.ttlCash = [_initial_cash, _initial_cash]
		self.ttlShare = [0,0]
		self.PV = [self.ttlCash[0],self.ttlCash[0]]
		self.inital_PV = deepcopy(self.PV)
		

		

	def LongOneShare(self,which):
		nowPrice = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now)
		tax = nowPrice*TRANSACTION_COST
		self.ttlCash[which] -= (nowPrice+tax)
		self.ttlShare[which] += 1

	def SellAndShortOne(self,which):
		nowPrice = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now)
		tax = nowPrice*(self.ttlShare[which]+1)*TRANSACTION_COST
		self.ttlCash[which] += ((self.ttlShare[which]+1)*nowPrice-tax)


		## should buy one back the next day
		tomoPrice[which] = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now+1)
		tax = tomoPrice*TRANSACTION_COST
		self.ttlCash[which] -= (tomoPrice+tax)

		# set Share to zero
		self.ttlShare[which] = 0

	def LongShares(self,which,nshares = 100):
		nowPrice = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now)
		tax = nowPrice*TRANSACTION_COST
		self.ttlCash[which] -= (nowPrice+tax)*nshares
		self.ttlShare[which] += nshares

	def SellShares(self,which,nshares=sys.maxint):
		nshares = min(self.ttlShare[which],nshares)

		nowPrice = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now)
		tax = nowPrice*(nshares)*TRANSACTION_COST
		self.ttlCash[which] += ((nshares)*nowPrice-tax)
		# set Share to zero
		self.ttlShare[which] -= nshares
		

	def TradeNext(self):
		today = self.now
		trendPred, trendPredX, nowD = self.dp.predictNext(stock=self.stocks, pred_date_count=today)
		trendReal = int(self.dp.getPriceByCount(stock=self.stocks, date_count=self.now+self.interval) >= self.dp.getPriceByCount(stock=self.stocks, date_count=self.now))

		# print "price:",self.dp.getPriceByCount(stock=self.stocks, date_count=self.now+self.interval) ,self.dp.getPriceByCount(stock=self.stocks, date_count=self.now)


		# print nowD,self.now
		# print trendPred,trendPredX,trendReal
		TRUEY.append(trendReal)
		PREDY.append(trendPred)

		if trendPred:
			# self.LongOneShare(which=0)
			self.LongShares(which=0)
		else:
			# self.SellAndShortOne(which=0)
			self.SellShares(which=0)

		if trendReal:
			# self.LongOneShare(which=1)
			self.LongShares(which=1)
		else:
			# self.SellAndShortOne(which=1)
			self.SellShares(which=1)

		self.now = today + self.interval
		# print self.now

	def getTotalROR(self):
		nowPrice = self.dp.getPriceByCount(stock=self.stocks, date_count=self.now)
		self.PV = [getPV(self.ttlCash[i], self.ttlShare[i], nowPrice) for i in range(2)]
		ttlROR = [float(self.PV[i]-self.inital_PV[i])/self.inital_PV[i] for i in range(2)]

		return ttlROR

def main():
	ZZZZ = Investor(_name='ZZZZ', _initial_virtual_shares=100, _start_date='2014-06-04', _stockcode=600050, _interval=7)

	while(ZZZZ.now<ZZZZ.maxcnt):
		sys.stdout.write('>')
		sys.stdout.flush()
		ZZZZ.TradeNext()
	print 
	print classification_report(TRUEY,PREDY)
	print "accu:",accuracy_score(TRUEY,PREDY)
	print ZZZZ.getTotalROR()




if __name__ == '__main__':
	main()







		