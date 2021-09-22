# Linear Regression 

Amazon_cloths sells cloths online. Customers come in to the store, have meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.

The company is trying to decide whether to focus their efforts on their mobile app experience or their website. 
Following is predict is analysis for this company

Just follow the steps below to analyze the customer data 

## Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

** Read in the Ecommerce Customers csv file as a DataFrame called customers.**


```python
customers = pd.read_csv('Ecommerce Customers')
```

```python
customers.head()
```

```python
customers.describe()
```

```python
customers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 8 columns):
    Email                   500 non-null object
    Address                 500 non-null object
    Avatar                  500 non-null object
    Avg. Session Length     500 non-null float64
    Time on App             500 non-null float64
    Time on Website         500 non-null float64
    Length of Membership    500 non-null float64
    Yearly Amount Spent     500 non-null float64
    dtypes: float64(5), object(3)
    memory usage: 31.3+ KB


## Data Analysis


```python
import seaborn as sns
```


```python
sns.jointplot(customers['Time on Website' ],customers['Yearly Amount Spent'])
```

** Do the same but with the Time on App column instead. **


```python
sns.jointplot(customers['Time on App'],customers['Yearly Amount Spent'])
```

**Let's explore these types of relationships across the entire data set **


```python
sns.pairplot(customers)
```

**Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**


```python
#Length of Membership
```

**Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **


```python
sns.lmplot(x='Yearly Amount Spent',y ='Length of Membership', data=customers)
```

## Training and Testing Data

Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **


```python
y = customers['Yearly Amount Spent']
```


```python
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
```

** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

## Training the Model

Now its time to train our model on our training data!

** Import LinearRegression from sklearn.linear_model **


```python
from sklearn.linear_model import LinearRegression
```

**Create an instance of a LinearRegression() model named lm.**


```python
lm = LinearRegression()
```

** Train/fit lm on the training data.**


```python
lm.fit(X_train,y_train)
```

    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



**Print out the coefficients of the model**


```python
print('Coefficients: \n', lm.coef_)
```

    Coefficients: 
     [ 25.98154972  38.59015875   0.19040528  61.27909654]


## Predicting Test Data
Now that we have fit our model, let's evaluate its performance by predicting off the test values!

** Use lm.predict() to predict off the X_test set of the data.**


```python
predictions = lm.predict(X_test)
```

** Create a scatterplot of the real test values versus the predicted values. **


```python
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
```
## Evaluating the Model

Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).

**Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. **


```python
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

    MAE: 7.22814865343
    MSE: 79.813051651
    RMSE: 8.93381506698


## Residuals

Let's quickly explore the residuals to make sure everything was okay with our data. 

**Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**


```python
sns.distplot((y_test-predictions),bins=50);
```

## Conclusion
We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.

**Do you think the company should focus more on their mobile app or on their website?**


*Mobile App*

We done it. Thank you
