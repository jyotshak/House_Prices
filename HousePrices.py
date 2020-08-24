#!/usr/bin/env python
# coding: utf-8

# In[297]:


import pandas as pd
import numpy as np
df1=pd.read_csv('logic_int.csv',index_col=0)
df2=pd.read_csv('train_str.csv',index_col=0)
ddf1=pd.read_csv('test_int.csv',index_col=0)
ddf2=pd.read_csv('test_str.csv',index_col=0)


# In[298]:


for i in range(0,44):
    for j in range(0,1459):
        ddf2.iloc[j,i]=str(ddf2.iloc[j,i])+str(i)


# In[299]:


for i in range(0,44):
    for j in range(0,1460):
        df2.iloc[j,i]=str(df2.iloc[j,i])+str(i)
df1   


# In[300]:


vocab=[]
for i in range(0,44):
    for j in range(0,1460):
        vocab.append(df2.iloc[j,i])
for i in range(0,44):
    for j in range(0,1459):
        vocab.append(ddf2.iloc[j,i])


# In[301]:


vocab=list(dict.fromkeys(vocab))
len(vocab)


# In[302]:


x_total=np.zeros(shape=(1460,291+28))


# In[303]:


for i in range(0,44):
    for j in range(0,1460):
        x_total[j,vocab.index(str(df2.iloc[j,i]))]=1


# In[304]:


for i in range(0,28):
    for j in range(0,1460):
        x_total[j,291+i]=float(df1.iloc[j,i])


# In[305]:


x_total


# In[306]:


y_total=np.zeros(shape=(1460,1))


# In[307]:


for i in range(0,1460):
    y_total[i]=df1.iloc[i,28]


# In[308]:


from matplotlib import pyplot as plt
import seaborn as sns
corr_matrix=df1.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True
f, ax = plt.subplots(figsize=(30, 40)) 
heatmap = sns.heatmap(corr_matrix, 
                      mask = mask,
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4, 
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1, 
                      vmax = 1,
                      annot = True,
                      annot_kws = {"size": 12})
#add the column names as labels
ax.set_yticklabels(corr_matrix.columns, rotation = 0)
ax.set_xticklabels(corr_matrix.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
heatmap.get_figure().savefig('heatmap.png', bbox_inches='tight')


# In[309]:



# lf1=pd.read_csv('logic_int.csv',index_col=0)
# lf2=pd.read_csv('logic_str.csv',index_col=0)
# lfy=pd.read_csv('price.csv',index_col=0)
# l_total=np.zeros(shape=(1460,int(28+188)))
# l_y=np.zeros(shape=(1460,1))
# for i in range(0,29):
#     for j in range(0,1460):
#         if(i!=28):
#             l_total[j,i]=float(lf1.iloc[j,i])
#         else:
#             l_y[j]=float(lf1.iloc[j,i])
# vocab_2=[]
# lf2
# for i in range(0,44):
#     for j in range(0,1460):
#         l_total[j,27+i]=float(vocab.index(df2.iloc[j,i]))


# In[310]:


# for i in range(0,32):
#     for j in range(0,1460):
#         lf2.iloc[j,i]=str(lf2.iloc[j,i])+str(i)
#         vocab_2.append(str(lf2.iloc[j,i]))
# vocab_2=list(dict.fromkeys(vocab_2))


# In[311]:


# len(vocab_2)


# In[312]:


# for i in range(0,32):
#     for j in range(0,1460):
#         l_total[j,28+vocab_2.index(str(lf2.iloc[j,i]))]=1
# temp=l_total


# In[313]:


temp


# In[314]:



# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# ss=RobustScaler()
# l_total=ss.fit_transform(l_total)
# l_train, l_test, l_y_train, l_y_test=train_test_split(l_total, l_y, test_size=0.2)
# from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import VotingRegressor,ExtraTreesRegressor, AdaBoostRegressor
# from sklearn.linear_model import LinearRegression
# import xgboost as xgb
# r1=MLPRegressor(activation='relu',solver='adam',learning_rate='adaptive',alpha=0.001, max_iter=1000)
# r2=ExtraTreesRegressor()
# r3=AdaBoostRegressor()
# r4=LinearRegression()
# r5=xgb.XGBRegressor()
# # regressor=VotingRegressor([('r1',r3),('r2',r1),('r3',r2)])
# regressor=r5
# regressor.fit(l_train,l_y_train.ravel())
# l_pred=regressor.predict(l_test)
# from sklearn import metrics
# import math
# error=metrics.mean_squared_error(l_pred,l_y_test)
# error=math.log10(math.sqrt(error))
# print(error)
# l_pred=regressor.predict(l_train)


# In[315]:


y_total.shape
x_total.shape
temp=x_total


# In[340]:


from sklearn.model_selection import train_test_split

x_total=ss.fit_transform(x_total)
x_train, x_test, y_train, y_test = train_test_split(x_total,y_total,test_size=0.2)


# In[ ]:





# In[350]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import ExtraTreesRegressor,AdaBoostRegressor,GradientBoostingRegressor,VotingRegressor,RandomForestRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
r1= ExtraTreesRegressor(n_estimators=100)
r2=MLPRegressor(tol=0.1,max_iter=5000)
r3=GradientBoostingRegressor()
r4=AdaBoostRegressor()
r5=xgb.XGBRegressor()
r6=RandomForestRegressor()
regressor=VotingRegressor([('a',r1),('b',r4),('c',r3),('d',r6)])
# regressor=r6
regressor.fit(x_train,y_train.ravel())


# In[351]:


import math
y_pred= regressor.predict(x_test)
from sklearn import metrics
error=metrics.mean_squared_error(y_test,y_pred)
error=math.log10(math.sqrt(error))
print(error)


# In[352]:


y_pred= regressor.predict(x_train)
error=metrics.mean_squared_error(y_train,y_pred)
error=math.log10(math.sqrt(error))
print(error)


# In[353]:


x_totaltest=np.zeros(shape=(1459,291+28))
for i in range(0,28):
    for j in range(0,1459):
        x_totaltest[j,291+i]=float(ddf1.iloc[j,i])
for i in range(0,44):
    for j in range(0,1459):
        x_totaltest[j,vocab.index(str(ddf2.iloc[j,i]))]=1


# In[354]:


x_totaltest=ss.transform(x_totaltest)
y_totalpred= regressor.predict(x_totaltest)


# In[355]:


y_totalpred.shape


# In[356]:


np.savetxt('result_1.csv',y_totalpred,delimiter=",")


# In[ ]:




