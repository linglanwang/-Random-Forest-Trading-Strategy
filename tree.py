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



def decisiontree_model(permno, train_range, test_range, gap, upthreshold, downthreshold):
	data = pd.read_csv('./data/'+str(permno)+'.csv', low_memory = False)
	x = data[['NORMBIDLO', 'NORMASKHI',  'NORMRET', 'NORMTRN', 'NORMOPENPRC', 'NORMsprtrn', 'PRCMAXGAP', 'PRCMINGAP', 'NORMPRC']]
	NEXTRET = data['PRC'].shift(-gap) / data['PRC'] - 1
	y = 0 + (NEXTRET > upthreshold) - (NEXTRET < -downthreshold)
	x_train = x.iloc[train_range[0]:train_range[1], :]
	y_train = y[train_range[0]:train_range[1]]

	train_tree = tree.DecisionTreeClassifier(class_weight='balanced')
	train_tree.fit(x_train, y_train)
	score = train_tree.score(x_train, y_train)
	
	x_test = x.iloc[test_range[0]:test_range[1], :]
	y_test = y[test_range[0]:test_range[1]]

	ret = NEXTRET[test_range[0]:test_range[1]]

	y_pre = train_tree.predict(x_test) 

	error = np.mean(np.logical_xor(y_test, y_pre))

	return permno, ret, y_pre, y_test, error, score


def foresttree_model(permno, train_range, test_range, gap, upthreshold, downthreshold):
	data = pd.read_csv('./data/'+str(permno)+'.csv', low_memory = False)
	x = data[['BIDLO', 'ASKHI',  'RET', 'TRN', 'OPENPRC', 'sprtrn', 'PRCMAXGAP', 'PRCMINGAP', 'PRC']]
	NEXTRET = data['PRC'].shift(-gap) / data['PRC'] - 1
	y = 0 + (NEXTRET > upthreshold) - (NEXTRET < -downthreshold)
	x_train = x.iloc[train_range[0]:train_range[1], :]
	y_train = y[train_range[0]:train_range[1]]
	
	train_tree = RandomForestClassifier(n_estimators=10, max_features='sqrt', max_depth=6, class_weight='balanced')
	train_tree.fit(x_train, y_train)
	score = train_tree.score(x_train, y_train)
	
	x_test = x.iloc[test_range[0]:test_range[1], :]
	y_test = y[test_range[0]:test_range[1]]

	ret = NEXTRET[test_range[0]:test_range[1]]

	y_pre = train_tree.predict(x_test) 

	error = 1-np.mean(y_test == y_pre)

	return permno, np.array(ret), np.array(y_pre), np.array(y_test), error, score


if __name__=="__main__":
	info = pd.read_csv('./count.csv')
	PERMNO = info['PERMNO']
	p = PERMNO[20]
	a,b,c,d,e,f = foresttree_model(p, [400,  700], [700, 740], 5, 0.03, 0.03)
	print(a,b,c,d,e,f)
	print(b.dot(c))


