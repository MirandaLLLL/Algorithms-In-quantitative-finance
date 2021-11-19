from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
import joblib
import sklearn

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from matplotlib import cm,pyplot as plt

import pymysql
conn=pymysql.connect(host='10.3.72.130',database='quant',user='root',password='stark12345+',port=3306)

sql='select trade_date,ts_code,open,close,high,low,pre_close, vol,amount from stock_daily'
data_use=pd.read_sql(sql,conn)


start_date = '2013-01-01'
end_date = '2017-12-31'
data_use["trade_date"] = pd.to_datetime(data_use["trade_date"])
data_use=data_use[(data_use['trade_date']>=start_date)&(data_use['trade_date']<=end_date)]



data_use1=data_use.set_index(["ts_code"])
close=data_use1['close']['000001.SZ']
high=data_use1['high']['000001.SZ'][5:]

low=data_use1['low']['000001.SZ'][5:]
volume=data_use1['vol']['000001.SZ'][5:]
money=data_use1['vol']['000001.SZ'][5:]


logreturn=(np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))[4:]
logreturn5=(np.log(np.array(close[5:]))-np.log(np.array(close[:-5])))
diffreturn = (np.log(np.array(high))-np.log(np.array(low)))
closeidx = close[5:]

X=np.column_stack([logreturn,diffreturn,logreturn5])

datelist= pd.to_datetime(data_use.trade_date[5:])

hmm=GaussianHMM(n_components = 6,covariance_type ='diag',n_iter = 5000).fit(X)
latent_states_sequence = hmm.predict(X)
len(latent_states_sequence)


data_use_2=data_use.drop_duplicates(subset=['trade_date'],keep='last')
datelist= pd.to_datetime(data_use_2.trade_date[:1208])

sns.set_style("white")
plt.figure(figsize = (15,8))
for i in range(hmm.n_components):
    state = (latent_states_sequence ==i)
    plt.plot(datelist[state],closeidx[state],'.',label='latent state %d'%i,lw =1)
    plt.legend()
    plt.grid(1)



data=pd.DataFrame({'datelist':datelist,'logreturn':logreturn,'state':latent_states_sequence}).set_index('datelist')
plt.figure(figsize=(15,8))

for i in range(hmm.n_components):
    state = (latent_states_sequence ==i)
    idx=np.append(0,state[:-1])
    data['state %d_return'%i] = data.logreturn.multiply(idx,axis = 0)
    plt.plot(np.exp(data['state %d_return'%i].cumsum()),label='latent state %d'%i)
    plt.legend()
    plt.grid(1)


buy = (latent_states_sequence ==1) + (latent_states_sequence ==2)
buy = np.append(0,buy[:-1])
sell = (latent_states_sequence ==0) +(latent_states_sequence ==3) + (latent_states_sequence ==4)+(latent_states_sequence ==5)
sell = np.append(0,sell[:-1])
data['backtest_return'] = data.logreturn.multiply(buy,axis=0)-data.logreturn.multiply(sell,axis=0)

plt.figure(figsize = (15,8))
plt.plot_date(datelist,np.exp(data['backtest_return'].cumsum()),'-',label='backtest result')
plt.legend()
plt.grid(1)
