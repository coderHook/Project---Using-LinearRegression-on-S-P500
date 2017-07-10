
# coding: utf-8

# In[51]:

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("sphist.csv")

df['Date'] = pd.to_datetime(df['Date'])

df_ordered = df.sort('Date', ascending = True)

print(df_ordered.head())


# In[71]:

# indicator to predict the price.
## We are using the shift() method at the end to adjust the dates and make everything move one day before to not get the day of today in the prediction.
mean_5_days = pd.rolling_mean(df_ordered['Close'], window = 5).shift(1)
mean_365_days = pd.rolling_mean(df_ordered['Close'], window = 365).shift(1)
ratio_mean_5vs365 = mean_5_days / mean_365_days

std_5_days = pd.rolling_std(df_ordered['Close'], window = 5).shift(1)
std_365_days = pd.rolling_std(df_ordered['Close'], window = 365).shift(1)
ratio_std_5vs365 = std_5_days / std_365_days

# Let's add two new indicators to see if we can reduce the error given. 
# As I consider Volume is also a good indicator in the stock market, we are going to add this indicator, for 5 days and a year.

mean_vol_5d = pd.rolling_mean(df_ordered['Volume'], window = 5).shift(1)
mean_vol_365d = pd.rolling_mean(df_ordered['Volume'], window = 365).shift(1)
ratio_mean_vol_5vs365 = mean_vol_5d / mean_vol_365d





# In[53]:

# Adding indicator about the standard deviation of the volume, for 5 days and 365

std_vol_5d = pd.rolling_std(df_ordered['Volume'], window=5).shift(1)
std_vol_365d = pd.rolling_std(df_ordered['Volume'], window=365).shift(1)
ratio_std_vol_5vs365 = std_vol_5d / std_vol_365d


# In[72]:

df_ordered['mean_5_days'] = mean_5_days
df_ordered['mean_365_days'] = mean_365_days
df_ordered['ratio_mean_5vs365'] = ratio_mean_5vs365
df_ordered['std_5_days'] = std_5_days
df_ordered['std_365_days'] = std_365_days
df_ordered['ratio_std_5vs365'] = ratio_std_5vs365
df_ordered['mean_vol_5d'] = mean_vol_5d
df_ordered['mean_vol_365d'] = mean_vol_365d
df_ordered['ratio_mean_vol_5vs365'] = ratio_mean_vol_5vs365

#adding the columns for the std vol
df_ordered['std_vol_5d'] = std_vol_5d
df_ordered['std_vol_365d'] = std_vol_365d
df_ordered['ratio_std_vol_5vs365'] = ratio_std_vol_5vs365


# In[73]:

print(df_ordered.head())


# In[74]:

# Lets remove any row that fall before 1951-01-03
new_df = df_ordered[df_ordered['Date'] >= datetime(year = 1951, month = 1, day = 3)]

new_no_df = new_df.dropna(axis=0)

train = new_no_df[new_no_df['Date'] < datetime(year=2013, month=1, day=1)]
test = new_no_df[new_no_df['Date'] >= datetime(year=2013, month=1, day=1)]
cols1 = ['mean_5_days', 'mean_365_days', 'ratio_mean_5vs365', 'std_5_days', 'std_365_days', 'ratio_std_5vs365']
cols2 = ['mean_5_days', 'mean_365_days', 'ratio_mean_5vs365', 'std_5_days', 'std_365_days', 'ratio_std_5vs365', 'mean_vol_5d', 'mean_vol_365d', 'ratio_mean_vol_5vs365']
cols3 = ['mean_5_days', 'mean_365_days', 'ratio_mean_5vs365', 'std_5_days', 'std_365_days', 'ratio_std_5vs365', 'mean_vol_5d', 'mean_vol_365d', 'ratio_mean_5vs365', 'std_vol_5d', 'std_vol_365d', 'ratio_std_vol_5vs365']    


# In[75]:

# Lets train our linearRegrssion model.
model = LinearRegression()
model.fit(train[cols1], train['Close'])

#Let's predict some data.
predictions = model.predict(test[cols1])

# Lets Calculate the error between predictions and test
mae1 = mean_absolute_error(test['Close'], predictions)

print(mae1)


# In[76]:

# Lets train our linearRegrssion model adding the columns mean volum for 5 and 365 days
model2 = LinearRegression()
model2.fit(train[cols2], train['Close'])

#Let's predict some data.
predictions2 = model2.predict(test[cols2])

# Lets Calculate the error between predictions and test
mae2 = mean_absolute_error(test['Close'], predictions2)

print(mae2)


# In[77]:

# Lets train our linearRegrssion model adding the columns std volum for 5 and 365 days
model2 = LinearRegression()
model2.fit(train[cols3], train['Close'])

#Let's predict some data.
predictions3 = model2.predict(test[cols3])

# Lets Calculate the error between predictions and test
mae3 = mean_absolute_error(test['Close'], predictions3)

print(mae3)


# - As we can see, adding the mean vol didnt change the error too much but adding aswell the standard deviation of this volume had a positive impact on the error. Now we have a better prediction. 
# 
# - Note: I tried to eliminate the mean vol and calculate the error just with the std vol and didnt change too much, seems that when we add more indicator the error is reduced.

# In[78]:

get_ipython().magic('matplotlib inline')

plt.figure(figsize=(20,10))

plt.plot(test['Date'], test['Close'], c="darkblue")
plt.plot(test['Date'], predictions, c="orange")
plt.show()


# - As we can observe, our predictions worked for this model pretty much ok, so we can rely in our model. Our MAE is about 16.1311 points, so its very accurate, in this case.

# In[80]:

# WE can use also the .score method in range [-1 , 1] to see how good is our prediction. 
# If it is close to 1 means that it is accurate. I can be even negative.

print(model.score(test[cols3], predictions3))
print(model.score(train[cols3], train['Close']))


# # Using more indicators to Reduce the error

# - Let's use some new indicators to reduce the error. 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



