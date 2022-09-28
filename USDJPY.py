# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#%% 把所有的文件拼接到一起
def v_concat_files (folder: str) -> pd.DataFrame:#垂直拼接表格 
    df_all = pd.DataFrame()
    for fn in os.listdir(folder):
        ffn = os.path.join(folder, fn) #全路径
        print(ffn)
        df_temp = pd.read_csv(ffn)
        df_all = df_all.append(df_temp)
    return df_all

df_1s = v_concat_files(
        'D:/DeutscheBank/week_1/USDJPY_1s/USDJPY',
        )


#%% 数据处理
df_1s['datetime'] = pd.to_datetime(df_1s['datetime'])
df_1s.head()
df_1s.dtypes
#按照每天为单位
df_1s['datetime'] = pd.to_datetime(df_1s['datetime']).dt.normalize()  
df_1s.head()
df_1s.dtypes


#创建中间汇率
df_1s['spread'] = df_1s['ap']-df_1s['bp']
df_1s['mean'] = (df_1s['ap']+df_1s['bp'])/2
df_1s.head()
df_1s.dtypes

df_1day = df_1s[['datetime','mean','spread']]
df_1day.rename(columns = {'mean':'price'})
df_1day.head()

#现在把相同天内的取平均值    
df_1day_mean = df_1day.groupby(['datetime']).mean()
df_1day_mean
df_1day_mean = df_1day_mean.rename(columns = {'mean':'mean_price'})
df_1day_mean

df_1day_mean.to_csv('df_1day_mean.csv')
#%% 读取文件
df_1day_mean = pd.read_csv('df_1day_mean.csv')
df_1day_mean['datetime'] = pd.to_datetime(df_1day_mean['datetime'])
#%%描述性统计及绘图
df_1day_mean.describe()



#日均价
plt.figure(figsize=(10,6))
plt.plot(df_1day_mean['mean_price'], c='r')
plt.title('mean_price', fontsize=18)
plt.xlabel('date', fontsize=12)
plt.ylabel('mean_price', fontsize=12) 

#日买卖价差
plt.figure(figsize=(10,6))
plt.plot(df_1day_mean['spread'], c='b')
plt.title('spread', fontsize=18)
plt.xlabel('date', fontsize=12)
plt.ylabel('spread', fontsize=12) 


#两张图结合
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('date')
ax1.set_ylabel('mean_price', color=color)
ax1.plot(df_1day_mean['mean_price'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('spread', color=color)  
ax2.plot(df_1day_mean['spread'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.show()


#%% momentum策略实施
#离散型市场单日收益
df_1day_mean['market_dis_returns'] = df_1day_mean['mean_price'].pct_change()

#离散型市场累计收益
df_1day_mean['market_dis_cum'] = (df_1day_mean['market_dis_returns']+1).cumprod()

# 仓位
df_1day_mean['position'] = np.sign(df_1day_mean['market_dis_returns'])

#策略单日收益
df_1day_mean['momentum_dis'] = df_1day_mean['position'].shift(1) * df_1day_mean['market_dis_returns'] 
# 策略累计收益
df_1day_mean['momentum_dis_cum'] = (df_1day_mean['momentum_dis']+1).cumprod()

#绘图
df_1day_mean [['momentum_dis_cum', 'market_dis_cum']].plot(
    figsize=(10,6), title = 'momentum_dis_cum')


#%% mean reversion策略

df_1day_mean['market_con_returns'] = np.log(
    df_1day_mean['mean_price']/df_1day_mean['mean_price'].shift(1))

df_1day_mean['sma'] = df_1day_mean['mean_price'].rolling(20).mean()

threshold = df_1day_mean['market_con_returns'].mean() + 1.5 * df_1day_mean['market_con_returns'].std()
df_1day_mean['market_con_returns'].dropna().plot(figsize=(10, 6), legend=True)
plt.axhline(threshold, color='r')
plt.axhline(-threshold, color='r')
plt.axhline(0, color='r')

df_1day_mean['position2'] = np.where(df_1day_mean['market_con_returns'] > threshold, 
                                    -1, np.nan)

df_1day_mean['position2'] = np.where(df_1day_mean['market_con_returns'] < -threshold, 
                                    1, df_1day_mean['position2'])

df_1day_mean['position2'] = np.where(df_1day_mean['market_con_returns'] * df_1day_mean['market_con_returns'].shift(1) < 0,
                                    0, df_1day_mean['position2'])

df_1day_mean['position2'] = df_1day_mean['position2'].ffill().fillna(0)

df_1day_mean['reversion'] = df_1day_mean['position2'].shift(1) * df_1day_mean['market_con_returns']
df_1day_mean[['market_con_returns', 'reversion']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))






#%% 补充
'''
#正常显示画图时出现的中文和负号
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False


# 设定1.5倍标准差为阈值，偏离绿色线的点将作为买入卖出信号
market_dis_returns = df_1day_mean['market_dis_returns']
market_dis_returns.plot(figsize=(14,6),label='日收益率')
plt.title('USDJPY日收益率',fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('日期', fontsize=12)
plt.axhline(market_dis_returns.mean(), 
            color='r',label='日收益均值')
plt.axhline(market_dis_returns.mean()+1.5*market_dis_returns.std(), 
            color='g',label='正负1.5倍标准差')
plt.axhline(market_dis_returns.mean()-1.5*market_dis_returns.std(), 
            color='g')
plt.legend()
ax = plt.gca()  
ax.spines['right'].set_color('none') 
ax.spines['top'].set_color('none')    
plt.show()

# 收益标准化
ret_20=market_dis_returns.rolling(20).mean()
std_20=market_dis_returns.rolling(20).std()
score=((market_dis_returns-ret_20)/std_20)
score.plot(figsize=(14,6),label='20日收益率标准化')
plt.title('USDJPY日收益标准化图',fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('',fontsize=12)
plt.axhline(score.mean(), color='r',label='日收益均值')
plt.axhline(score.mean()+1.5*score.std(), color='g',label='正负1.5倍标准差')
plt.axhline(score.mean()-1.5*score.std(), color='g')
plt.legend()
ax = plt.gca()  
ax.spines['right'].set_color('none') 
ax.spines['top'].set_color('none')    
plt.show()
'''




