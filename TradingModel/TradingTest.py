# -*- coding:utf-8 -*-  
import sys
sys.path.append('../System/')
from Basic import Stock
from copy import deepcopy
from Tool import DataProcessor, default_divide_ratio
from helper import *
from sklearn.metrics import classification_report, accuracy_score, f1_score
from progressbar import *
import random




TRANSACTION_COST = .003

# TRUEY = []
# PREDY = []

STOCK_POOL = [600030]
# STOCK_POOL = [600030, 600570, 600051, 600401, 600691, 600966, 600839]
# STOCK_POOL = [600210, 600487, 600598, 600419, 600572, 600718, 600756, 600536, 600776]



def getPV(cash,share,price):
	return cash + share*price

class Investor:
	def __init__(self,_name='ZZZZ', _initial_virtual_shares=1000, _start_date='2010-06-04', _end_date=None, _stockcode=600050, _interval=10,_train_batch_size = 100):
		self.name = _name
		self.now = 0
		self.interval = _interval
		self.train_batch_size = _train_batch_size
		self.TRUEY = []
		self.PREDY = []
		self.TRAINERROR = []

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
		self.stocks = Stock(SN=_stockcode, start_date=_start_date, interval=_interval,granularity=_interval, granu_count=10)
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
		use_NN = False
		trendPredY, trendPred_prob, trendLabelY, nowD , trainError= self.dp.predictNext(stock=self.stocks, pred_date_count=today, train_batch_size=self.train_batch_size, use_NN=use_NN)
		trendReal = int(self.dp.getPriceByCount(stock=self.stocks, date_count=self.now+self.interval) >= self.dp.getPriceByCount(stock=self.stocks, date_count=self.now))

		self.TRUEY.append(trendReal)
		self.PREDY.append(trendPredY)
		self.TRAINERROR.append(trainError)

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

def backtestHistory(_initial_virtual_shares, _start_date, _stockcode, _interval,_train_batch_size = 100):
	ZZZZ = Investor(_name='ZZZZ', _initial_virtual_shares=_initial_virtual_shares, _start_date=_start_date, _stockcode=_stockcode, _interval=_interval,_train_batch_size = _train_batch_size)
	total = ZZZZ.maxcnt-ZZZZ.now
	# pbar = ProgressBar(widgets=[' ', AnimatedMarker(), 'Predicting: ', Percentage()], maxval=total).start()
	while ZZZZ.now < ZZZZ.maxcnt:
	    # pbar.update(ZZZZ.now)
	    # time.sleep(0.01)
	    ZZZZ.TradeNext(use_NN=False)
	# pbar.finish()

	print
	print classification_report(ZZZZ.TRUEY, ZZZZ.PREDY)
	f1 = f1_score(ZZZZ.TRUEY, ZZZZ.PREDY)
	accuracy = accuracy_score(ZZZZ.TRUEY, ZZZZ.PREDY)
	print "accuracy:", accuracy
	print "f1: ",f1
	predROR = ZZZZ.getTotalROR()[0]
	realROR = ZZZZ.getTotalROR()[1]
	assert not (realROR == 0)
	print 'pred ROR:', predROR, '%', '\t|\treal ROR:', realROR, '%'

	return predROR, realROR, f1, accuracy, total, ZZZZ.TRAINERROR


def StockSelection(stock_pool):
	predict_list = {}
	over_list = {}
	f1_list = {}
	accuracy_list = {}

	for _code in stock_pool:
		print "--------------------------------------------------------------------------------"
		(predR, realR, f1, accuracy,ttl) = backtestHistory(_initial_virtual_shares=100, _start_date='2014-06-04', _stockcode=_code, _interval=10)
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

def StockSearch(intvl = 5,cores = 4):

	import os
	import threading


	foldername = "./result("+str(intvl)+"day)3of4restrest/"

	if not os.path.exists(foldername):
		os.mkdir(foldername)

	with open("stockPool.li","r") as f1:
		stock_pool = [eval(item) for item in eval(f1.read())]
	# stock_pool = [600210, 600487, 600598, 600419, 600572, 600718, 600756, 600536, 600776]
	stock_pool = stock_pool[int(len(stock_pool)*0.0)+704:int(len(stock_pool)*0.75)]
	# stock_pool = stock_pool[::-1]
	# stock_pool = stock_pool[128:int(len(stock_pool)*0.33)]


	# predict_list = {}
	# over_list = {}
	# f1_list = {}
	# accuracy_list = {}
	def thd(i):
		name = i
		fw = open(foldername+str(name)+'.tmp','w+')
		fw.close()
		# fw = open(foldername+str(i)+'.tmp','a')

		while(i<len(stock_pool)):
			try:
				fw = open(foldername+str(name)+'.tmp','a')
				(predR, realR, f1, accuracy, ttl) = backtestHistory(_initial_virtual_shares=100, _start_date='2013-06-05', _stockcode=stock_pool[i], _interval=intvl)
				fw.write("'"+str(stock_pool[i])+','+str(predR)+','+str(realR)+','+str(float(predR/realR))+','+str(f1)+','+str(accuracy)+","+str(ttl)+'\n')
				fw.close()
			except:
				pass
			i+=cores
			

	# TASK = []
	# for i in range(cores):
	# 	TASK.append(threading.Thread(target = thd,args = (i,)))
	# for t in TASK:
	# 	t.start()
	# for t in TASK:
	# 	t.join();
	thd(0)

	fres = open(foldername+'res_'+str(intvl)+"day.csv","w+")
	fres.write('code,PredReturnRate,RealReturnRate,OverReturnRate,f1,accuracy,ttldays\n');
	for i in range(cores):
		ftmp = open(foldername+str(i)+'.tmp','r')
		fres.write(ftmp.read())
		ftmp.close()
	fres.close()








	# for _code in stock_pool:
	# 	(predR, realR, f1, accuracy) = backtestHistory(_initial_virtual_shares=100, _start_date='2013-06-02', _stockcode=_code, _interval=intvl)
	# 	predict_list[_code] = predR
	# 	over_list[_code] = float(predR/realR)
	# 	f1_list[_code] = f1
	# 	accuracy_list[_code] = accuracy

	# predict_list = sort_dict(predict_list)
	# over_list = sort_dict(over_list)
	# f1_list = sort_dict(f1_list)
	# accuracy_list = sort_dict(accuracy_list)

	# print "Predict Return rate: ", predict_list
	# print "Over Return rate: ", over_list
	# print "f1-score rank: ", f1_list
	# print "accuracy rank: ", accuracy_list

	# DIR_Storage = '../Results/'
	# save(str(predict_list), DIR_Storage+"predict_list.list")
	# save(str(over_list), DIR_Storage+"over_list.list")
	# save(str(f1_list), DIR_Storage+"f1_list.list")
	# save(str(accuracy_list), DIR_Storage+"accuracy_list.list")
def learningCurve():
	f_lc = open('learningCurveDT.csv','w+')
	for trainsize in [20,30,50,80,100,120,150,200,300]:
		(predR, realR, f1, accuracy, ttl, train_error) = backtestHistory(_initial_virtual_shares=100, _start_date='2013-06-05', _stockcode=600530, _interval=10,_train_batch_size = trainsize )
		print 'trainsize: ',trainsize,' train_error: ',sum(train_error)/len(train_error),' test_error:',1-accuracy
		f_lc.write(str(trainsize)+ "," +str(sum(train_error)/len(train_error))+","+str(1-accuracy)+"\n")
	f_lc.close()

def selectDays():
	for days in [5,15,20,25,35,40,45]:
		(predR, realR, f1, accuracy, ttl, train_error) = backtestHistory(_initial_virtual_shares=100, _start_date='2013-06-05', _stockcode=600530, _interval=days )
		# print 'trainsize: ',trainsize,' train_error: ',sum(train_error)/len(train_error),' test_error:',1-accuracy
		# f_lc.write(str(trainsize)+ "," +str(sum(train_error)/len(train_error))+","+str(1-accuracy)+"\n")


if __name__ == '__main__':
	# StockSelection(stock_pool=STOCK_POOL)
	# StockSearch(intvl = 10,cores = 1)
	selectDays()






