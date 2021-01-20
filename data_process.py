import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import fsolve
from scipy.stats import norm
from sympy import *

def norm_data(data, columns, window):
	for column in columns:
		data[column+'MEAN'] = data[column].rolling(window).mean()
		data[column+'STD'] = data[column].rolling(window).std()
		data[column+'MAXGAP'] = (data[column].rolling(window).max() - data[column]) / data[column]
		data[column+'MINGAP'] = -(data[column].rolling(window).min() - data[column]) / data[column]
		data['NORM'+column] = (data[column] - data[column+'MEAN']) / data[column+'STD']

rawdata = pd.read_csv('2016-2019.csv', low_memory = False)
PERMNO = np.unique(rawdata['PERMNO'])
df = pd.DataFrame([], columns = ['PERMNO', 'TICKER', 'Count'])
for p in PERMNO:
	data = rawdata[rawdata['PERMNO'] == p]
	if len(data) < 1006: continue
	data.reset_index(inplace = True, drop = True)
	data['RET'] = data['RET'].astype(float)
	data['PRC'] = data['PRC'].abs()
	data['PRC'] = data['PRC'] / data['CFACPR']
	data['BIDLO'] = data['BIDLO'] / data['CFACPR']
	data['ASKHI'] = data['ASKHI'] / data['CFACPR']
	data['OPENPRC'] = data['OPENPRC'] / data['CFACPR']
	data['SHROUT'] = data['SHROUT'] * data['CFACSHR']
	data['TRN']  = data['VOL'] / (1000 * data['SHROUT'])
	data['logBIDLO'] = np.log(data['BIDLO'])
	data['logASKHI'] = np.log(data['ASKHI'])
	data['logOPENPRC'] = np.log(data['OPENPRC'])
	data['logPRC'] = np.log(data['PRC'])
	norm_data(data, ['PRC', 'ASKHI', 'BIDLO', 'OPENPRC', 'RET', 'TRN', 'sprtrn'], 252)
	data = data.loc[252: ,:]
	data.reset_index(inplace = True, drop = True)
	data.to_csv('./data/'+str(p)+'.csv')
	tick = data.iloc[0]['TICKER']
	count = len(data)
	df.loc[len(df)] = [p, tick, count]
df.to_csv('count.csv')
