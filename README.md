# Predicting the S&P500 using LinearRegression()

### In this project I will be using LinearRegression to predict th price of the S&P500 index.
### To do so:
  - We are going two separate the dataframe in two sets.
      - Train --> Will be rows with date from 1950 to 2013
      - Test --> Rows with dates from 2013 to 2015

  We have to be careful in the date that we select for our indicator to not select the day that we want to predict, cause in this case our indicator will not reflect properly what is happening in the rel world because in real world we do not know which is the closing price for actual day.

## Indicators used in this model:
  - Mean price for 5 days, 365 days and the ratio of both
  - Standard deviation for 5, 365 days and the ratio between them.

  - Mean Volume for 5, 365d and the ratio
  - Std Vol for 5, 365d and the ratio

  ## Indicators that we can use:

    - The year component of the date.
    - The ratio between the lowest price in the past year and the current price.
    - The ratio between the highest price in the past year and the current price.
    - The year component of the date.
    - The month component of the date.
    - The day of week.
    - The day component of the date.
    - The number of holidays in the prior month.

Finally we will print a plot on we can see how accurate is our prediction with the data given on the test.
Also, We will print the different errors that we got from the indicator to see how accurate are our prediction, and which ones worked and which ones does not.
