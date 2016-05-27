from Basic import Stock, MarketPortfolio
from Tool import DataProcessor
import numpy as np
import pandas as pd
import os

MP_filename = "MarketPortfolio.mdl"

def play(self):
	cnt = 0
	error = 0
	val = []
	for i in xrange(600000,601000):
		filename = "../data/"+str(i)+"_ss.csv"
		if os.path.exists(filename):
			cnt += 1
			try:
				raw = np.array(pd.read_csv(filename))
			except Exception, e:
				error += 1
				# print Exception,":",e
				continue
			vol = np.array(raw[:, 5])
			zero = 0
			for x in range(len(vol)):
				if vol[x]==0:

					zero += 1
			val.append(zero)
			print "SN: ", i, "- have zero = ", zero


	print min(val)
	print max(val)

	print cnt
	print error




def Train():
	StockPool = [600050, 600401, 600691, 600966, 600839]
	# 
	pass

if __name__ == '__main__':

	stk = Stock(600050, '2005-06-02', 1)
	dp = DataProcessor(stk, 1)

	print stk._start, stk._end
	print stk._index

	print ' ---- '
	print dp.X_raw.shape
	print dp.date_raw

	# print dp.date_raw
	# print dp.X_raw[:][0]
