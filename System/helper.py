# -*- coding:utf-8 -*-  
import os
from operator import itemgetter


def sort_dict(Dict):
	"""
		sort dictionary by values

	"""
	return sorted(Dict.items(), key=itemgetter(1), reverse=True)


def load(filename):
	f = file(filename, 'r+')
	result = eval(f.read())
	f.close()
	return result


def save(item, filename):
	f = file(filename, 'w+')
	f.write(str(item))
	f.close()


if __name__ == '__main__':
	pass

