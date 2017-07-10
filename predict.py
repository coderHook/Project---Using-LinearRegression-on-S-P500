import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("sphist.csv")

df['Date'] = pd.to_datetime(df['Date'])

df_ordered = df.sort('Date', ascending = True)

# indicator to predict the price.
## We are using the shift() method at the end to adjust the dates and make everything move one day before to not get the day of today in the prediction.
mean_5_days = pd.rolling_mean(df_ordered['Close'], window = 5).shift(1)
mean_365_days = pd.rolling_mean(df_ordered['Close'], window = 365).shift(1)
ratio_mean_5vs365 = mean_5_days / mean_365_days

std_5_days = pd.rolling_std(df_ordered['Close'], window = 5).shift(1)
std_365_days = pd.rolling_std(df_ordered['Close'], window = 365).shift(1)
ratio_std_5vs365 = std_5_days / std_365_days

df_ordered['mean_5_days'] = mean_5_days
df_ordered['mean_365_days'] = mean_365_days
df_ordered['ratio_mean_5vs365'] = ratio_mean_5vs365
df_ordered['std_5_days'] = std_5_days
df_ordered['std_365_days'] = std_365_days
df_ordered['ratio_std_5vs365'] = ratio_std_5vs365

# Lets remove any row that fall before 1951-01-03
new_df = df_ordered[df_ordered['Date'] >= datetime(year = 1951, month = 1, day = 3)]

new_no_df = new_df.dropna(axis=0)

train = new_no_df[new_no_df['Date'] < datetime(year=2013, month=1, day=1)]
test = new_no_df[new_no_df['Date'] >= datetime(year=2013, month=1, day=1)]
cols = ['mean_5_days', 'mean_365_days', 'ratio_mean_5vs365', 'std_5_days', 'std_365_days', 'ratio_std_5vs365']
    
# Lets train our linearRegrssion model.
model = LinearRegression()
model.fit(train[cols], train['Close'])

#Let's predict some data.
predictions = model.predict(test[cols])

# Lets Calculate the error between predictions and test
mae = mean_absolute_error(test['Close'], predictions)

print(mae.head)
plt.plot(test['Close'])
plt.show()

