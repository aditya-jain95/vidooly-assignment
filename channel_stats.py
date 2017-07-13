
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# In[2]:

df = pd.read_csv('oct_march.csv')
df.head()


# In[3]:

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('oct_march.csv', parse_dates=['date'], index_col='date',date_parser=dateparse)
print data.head()


# In[4]:

data['chid'].value_counts()


# In[5]:

#CHOSE the following Channel with ID:UC6ROKPXrnzfhNYST1w
channel = data[data.chid=='UC6ROKPXrnzfhNYST1w']
channel.drop('chid',axis=1,inplace=True)
len(channel)


# In[55]:

channel.tail()


# In[147]:

plt.subplot(311)
plt.plot(channel['views'], label='views')
plt.legend(loc='best')

plt.subplot(312)
plt.plot(channel['subscriber'], label='subscriber')
plt.legend(loc='best')

plt.subplot(313)
plt.plot(channel['videoscount'], label='video count')
plt.legend(loc='best')


# In[8]:

tsv = channel['views']
tss = channel['subscriber']
tsvc = channel['videoscount']


# In[9]:

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=30)
    rolstd = pd.rolling_std(timeseries, window=30)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput


# In[148]:

#Now we will test for stationarity 


# In[11]:

test_stationarity(tsv)
test_stationarity(tss)
test_stationarity(tsvc)


# In[12]:

tsv_log = np.log(tsv)
moving_avg = pd.rolling_mean(tsv_log,30)
tsv_moving_avg_diff = tsv_log - moving_avg
tsv_moving_avg_diff.head(32)


# In[13]:

tsv_moving_avg_diff.dropna(inplace=True)
test_stationarity(tsv_moving_avg_diff)


# In[27]:

tsv_log_diff = tsv_log - tsv_log.shift()
tsv_log_diff.dropna(inplace=True)
test_stationarity(tsv_log_diff)


# In[102]:

tss_log = np.log(tss)
tss_log_diff = tss_log - tss_log.shift(1) 
tss_log_diff.dropna(inplace=True)
test_stationarity(tss_log_diff)


# In[103]:

tsvc_log = np.log(tsvc)
moving_avg = pd.rolling_mean(tsvc_log,30)
tsvc_moving_avg_diff = tsvc_log - moving_avg
tsvc_moving_avg_diff.dropna(inplace=True)
test_stationarity(tsvc_moving_avg_diff)


# In[116]:

tsvc_log_diff = tsvc_log - tsvc_log.shift()
tsvc_log_diff.dropna(inplace=True)
test_stationarity(tsvc_log_diff)


# In[19]:

#tsv_log_diff
#tsvc_log_diff
#tss_log_decompose


# In[20]:

import warnings
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.8f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.8f' % (best_cfg, best_score))

# load dataset

# evaluate parameters
p_values = [0, 1, 2]
d_values = [0,1]
q_values = range(0, 2)
warnings.filterwarnings("ignore")
evaluate_models(tsv_log, p_values, d_values, q_values)


# In[76]:

from statsmodels.tsa.statespace.sarimax import SARIMAX

mod1 = SARIMAX(tsv_log, order=(2,1,1))
res1 = mod1.fit()

mod2 = SARIMAX(tsv_log, order=(2,1,1))
res2 = mod2.filter(res1.params)
pred_v = res2.forecast(91)


# In[77]:

pred_v = np.exp(pred_v)


# In[80]:

pred_v = np.round_(pred_v)


# In[38]:

p_values = [0, 1, 2]
d_values = [0,1]
q_values = range(0, 2)
warnings.filterwarnings("ignore")
evaluate_models(tsvc_log, p_values, d_values, q_values)


# In[127]:

mod1 = SARIMAX(tsvc_log, order=(1,1,0))
res1 = mod1.fit()

mod2 = SARIMAX(tsvc_log, order=(1,1,0))
res2 = mod2.filter(res1.params)
pred_vc = res2.forecast(91)


# In[128]:

pred_vc = np.exp(pred_vc)


# In[137]:

pred_vc = np.round_(pred_vc)
pred_vc


# In[94]:

p_values = [0, 1, 2]
d_values = [0,1]
q_values = range(0, 2)
warnings.filterwarnings("ignore")
evaluate_models(tss_log, p_values, d_values, q_values)


# In[96]:

mod1 = SARIMAX(tss_log, order=(1,1,0))
res1 = mod1.fit()

mod2 = SARIMAX(tss_log, order=(1,1,0))
res2 = mod2.filter(res1.params)
pred_s = res2.forecast(91)


# In[104]:

pred_s = np.round_(np.exp(pred_s))


# In[107]:

submission = pd.DataFrame()
chid = 'UC6ROKPXrnzfhNYST1w'
submission['chid'] = [chid] * 91


# In[138]:

submission['views'] = pred_v.values.astype(int)
submission['subscriber'] = pred_s.values.astype(int)
submission['videoscount'] = pred_vc.values.astype(int)


# In[139]:

submission.head()


# In[142]:

submission['date']=pd.date_range(start='2017-04-01',end='2017-06-30',freq='D')


# In[145]:

submission.head()


# In[146]:

submission.to_csv('channel_stats.csv',index=False)


# In[ ]:



