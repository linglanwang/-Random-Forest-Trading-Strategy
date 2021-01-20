import datetime as dt

import pandas as pd
import numpy as np
import tushare as ts
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from multiprocessing import Pool
from tree import decisiontree_model, foresttree_model


def trading(permno, train_range, test_range, gap, upthreshold, downthreshold, return_info = True):
	permno, ret, y_pre, y_test, error, score = decisiontree_model(permno, train_range, test_range, gap, upthreshold, downthreshold)
	trade_signal = y_pre
	trade_signal = np.nan_to_num(trade_signal)
	rets = ret * trade_signal
	ret_cum = np.cumsum(rets)
	total_ret = trade_signal.dot(ret)
	if return_info == True: 
		df = pd.DataFrame(np.array([rets, ret_cum]).T,  columns = ['return', 'cumulative return'])
		df.to_csv('./result/'+str(permno)+'.csv')
		return permno, score, error, total_ret
	return  permno, ret, ret_cum, total_ret, error, score


def trading_forest(permno, train_range, test_range, gap, upthreshold, downthreshold, return_info = True):
	permno, ret, y_pre, y_test, error, score = foresttree_model(permno, train_range, test_range, gap, upthreshold, downthreshold)
	trade_signal = y_pre
	trade_signal = np.nan_to_num(trade_signal)
	rets = ret * trade_signal
	ret_cum = np.cumsum(rets)
	total_ret = trade_signal.dot(ret)
	if return_info == True: 
		df = pd.DataFrame(np.array([rets, ret_cum]).T,  columns = ['return', 'cumulative return'])
		df.to_csv('./result_forest/'+str(permno)+'.csv')
		return permno, score, error, total_ret
	return  permno, ret, ret_cum, total_ret, error, score


if __name__=="__main__":
	info = pd.read_csv('./count.csv')
	PERMNO = info['PERMNO']
	result = []
	for  p in PERMNO:
		try:

			permno, score, error, total_ret = trading(p, [400,  700], [700, 740], 5, 0.03, 0.03)
			result.append([permno, score, error, total_ret])
		except ValueError:
			pass
		continue
	err_ret = pd.DataFrame(result, columns = ['PERMNO', 'score', 'error', 'total return'])
	err_ret.sort_values(by = 'error', ascending = True, inplace = True, ignore_index = True)
	err_ret.to_csv('error_ret.csv')


	result = []
	for  p in PERMNO:
		try:

			permno, score, error, total_ret = trading_forest(p, [400,  700], [700, 740], 5, 0.03, 0.03)
			result.append([permno, score, error, total_ret])
		except ValueError:
			pass
		continue
	err_ret = pd.DataFrame(result, columns = ['PERMNO', 'score', 'error', 'total return'])
	err_ret.sort_values(by = 'error', ascending = True, inplace = True, ignore_index = True)
	err_ret.to_csv('error_ret'+'_forest'+'.csv')




