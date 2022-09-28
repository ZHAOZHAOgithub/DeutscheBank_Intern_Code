# -*- coding: utf-8 -*-
"""
csi300和上证50两个指数（每日收盘价）000300  000016
策略：pair trading
流程：价格相关系数检验（一般用ln价格）
    价格平稳性检验（非平稳则进行一阶差分）
    对两组数据进行协整性检验    


"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm  #协整检验
from statsmodels.tsa.stattools import adfuller as ADF #导入ADF函数
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#%% 获取数据

df_50 = pd.read_csv('000016.csv',encoding='gbk')
df_300 = pd.read_csv('399300.csv',encoding='gbk')

#%% 清理数据
df_50.dtypes
df_50 = df_50[['日期','收盘价']]
df_50 = df_50.rename({'日期':'trade_date', '收盘价': 'price_50'}, axis = 1)
df_50['trade_date'] = pd.to_datetime(df_50['trade_date'])
df_50.dtypes

df_300 = df_300[['日期','收盘价']]
df_300 = df_300.rename({'日期':'trade_date', '收盘价': 'price_300'}, axis = 1)
df_300['trade_date'] = pd.to_datetime(df_300['trade_date'])
df_300.dtypes
df_300 = df_300.head(2814)
df_50 = df_50.head(2814)

df = pd.merge(df_50, df_300, on = 'trade_date')
#%% 检验
#%%%价差

df['spread'] = df['price_300'] - df['price_50']
spread_mean = df['spread'] .mean()
df['spread'].plot(label='spread')
plt.axhline(y = spread_mean, color='black')

#一般不直接用价差，而用后面模拟出的model的残差，因为残差具有平稳性，可以更好地运用均值回归
#%%%相关系数

np.corrcoef(df['price_300'], df['price_50'])
#发现相关系数达到0.9815

#%%%平稳性检验


def adf_test (S:float):  #原数据adf检验
    adf = ADF(S)
    result = pd.Series(adf[0:4],
                   index=[
                       'Test Statistic', 'p-value', 'Lags Used',
                       'Number of Observations Used'
                   ])
    return result



# 上证50数据
df['log_50'] = np.log(df['price_50'])
adf_test(df['log_50']) #对价格使用对数

#ADF检验的P值一般小于0.05时，拒绝原假设。实际的p-value大于0.05，无法拒绝原假设，所以应该认为对数价格为非平稳序列。
#对数价格不平稳，进行一阶差分

diff_1_50 = df['log_50'].diff()[1:]  #对一阶差分之后的数据作ADF检验
adf_test(diff_1_50)
#p-vlaue<0.05,一阶差分为平稳序列，上证50对数价格再作一阶差分之后的序列数据为一阶单整序列。



#沪深300数据
df['log_300'] = np.log(df['price_300'])
adf_test(df['log_300'])#原数据
diff_1_300 = df['log_300'].diff()[1:]#一阶差分
adf_test(diff_1_300)

#得出结论：沪深300和上证50的log值是一阶单整序列

#%%% 协整检验
model = sm.OLS(df['log_50'],sm.add_constant(df['log_300']))
results = model.fit()
print(results.summary())


#对回归残差进行平稳性检验
alpha = results.params[0] #提取回归截距
beta = results.params[1] # 提取回归系数
s = df['log_50'] - beta * df['log_300'] - alpha  #求残差序列
adf_test(s) 
# 此时p=0.11>0.05， 不平稳
#取一阶差分
adf_test(s.diff()[1:])
# 此时p<0.05, 残差为一阶单整

# 计算协整方程中残差序列的均值、方差
me = np.mean(s)
sd = np.std(s)
#%% 交易策略
#%%% 价差序列
plt.figure(figsize=(12,6))
s.plot()
plt.title('价差(残差)序列 上证50 = 1.0272沪深300 - 0.5852',loc='center', fontsize=16)
plt.axhline(y = me, color = 'black')
plt.axhline(y = me+1.5*sd, color = 'blue', ls = '-', lw = 2)
plt.axhline(y = me-1.5*sd, color = 'blue', ls = '-', lw = 2)
plt.axhline(y = me+2*sd, color = 'green', ls = '--', lw = 2.5)
plt.axhline(y = me-2*sd, color = 'green', ls = '--', lw = 2.5)
plt.show()
#开仓阈值设置为窗口内数据的1.5倍标准差，止损设置为2倍标准差
#%%% 设置交易信号
#上证50 = 1.0272沪深300 - 0.5852
#考虑持有上证50-1.0270沪深300
open = 1.5*sd
stop = 2*sd
#如果残差从下方向上穿过蓝线，则卖出1份上证50，买入1.027份沪深300，此时交易信号为-1

price_50 = df['price_50']
price_300 = df['price_300']
hold = False  #假设一开始没有持仓
profit_list = []  #存放收益
profit_sum = 0
hold_price_50 = 0  #持仓价格
hold_price_300 = 0 #持仓价格
position = 0 #-1为卖出上证50，买入沪深300
for i in range(len(price_50)):
    if hold == False:  #如果手上没有持仓
        if s[i] >= open:    #残差大于开仓线，仓位-1
            hold_price_50 = price_50[i]
            hold_price_300 = price_300[i]
            position = -1
            hold = True
        elif s[i]<= -open:  #残差小于负开仓线，仓位1
            hold_price_50 = price_50[i]
            hold_price_300 = price_300[i]
            position = 1
            hold = True
    else: #如果手上有持仓
        if  position == -1 and s[i] >= stop  : #仓位为-1，且触及上止损线时，强制平仓
            profit = (hold_price_50 - price_50[i]) + 1.027*(price_300[i] - hold_price_300)
            profit_sum += profit
            position = 0
            hold = False
        if  position == 1 and s[i] <= -stop : #仓位为1， 且触及下止损线，强制平仓
            profit = (price_50[i] - hold_price_50) + 1.027*(hold_price_300 - price_300[i])
            profit_sum += profit
            hold_state = 0
            hold = False
        if  position == -1 and s[i] <= 0: #仓位为-1，且残差触及0时，平仓。
            profit = (hold_price_50 - price_50[i]) + 1.027*(price_300[i] - hold_price_300)
            profit_sum += profit
            hold_state = 0
            hold = False
        if  position == 1 and s[i] >= 0: #仓位为1， 且残差触及0时，平仓。认为上证50已经涨够了
            profit = (price_50[i] - hold_price_50) + 1.027*(hold_price_300 - price_300[i])
            profit_sum += profit
            hold_state = 0
            hold = False
        profit_list.append(profit_sum)

plt.plot(profit_list)
