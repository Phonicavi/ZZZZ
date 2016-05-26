from Basic import Stock, MarketPortfolio
import os
import numpy as np
import pandas as pd

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
	pass


if __name__ == '__main__':
	if not os.path.exists(MP_filename):
		mkt = MarketPortfolio()
	pass

	stk = Stock(600050)
	# print stk.beta

