# Price_Prediction_model
Complete Machine Learning Algorithm for Housing Price Prediction model by using Regressor as a Supervised model


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

## Importing housing data set and stroring it into housing variable 

housing = pd.read_csv('data.csv')

housing.head()    # Used to check the appreiance of our dataset in dataframe

housing['CHAS'].value_counts()    # to check perticular feature values counting 

housing.describe()    # to check any null data set values are there or not in our data set

housing.info()    # to varify if any null value are there in data set or not

## plotting histogram plot to check the data set visulization

# housing.hist(bins=50 , figsize=(10,2))
# plt.show()    # Data in the form of histogram we do not want to consume more space henve we did comment out this line of code

## Splitting our data set into train and test set

## option 1 -  by creating own function 

def train_test_split(data, test_ratio):
    np.random.seed(42)    # to fix our random shuffing in first iteration only
    shuffle = np.random.permutation(len(data))    # shuffle variable created which used to dp shuffing of our data
    test_set_size = int(len(data)*test_ratio)     # created test set size 
    test_indices = shuffle[:test_set_size]
    train_indices = shuffle[test_set_size:]
    
    return data.iloc[train_indices] , data.iloc[test_indices]

train_set , test_set = train_test_split(housing,0.2)

print(f"The numbers of rows in train set is :{len(train_set)} \n The number of rows in test set : {len(test_set)}")

train_set.shape

test_set.shape

## Option 2 - Skicit learn inbuild function 

from sklearn.model_selection import train_test_split

train_set , test_set = train_test_split(housing , test_size= 0.2 , random_state=42)

print(f"The numbers of rows in train set is :{len(train_set)} \n The number of rows in test set : {len(test_set)}")

## Stratified Shuffle Split 

Used for splitting equal population value of perticular feature into each train and test data set

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits= 1 , test_size= 0.2 , random_state=42)

for train_index , test_index in split.split(housing, housing['CHAS']):   ## Considering Chas column is very important feature
    strt_train_set = housing.loc[train_index]
    strt_test_set = housing.loc[test_index]

strt_test_set['CHAS'].value_counts()

strt_train_set['CHAS'].value_counts()

## Checking for correlation between columns with respect to Price MEDV columns

corr_matrix = housing.corr()   # created variable with pandas corr fucntion to check correlation between two variables 

corr_matrix['MEDV'].sort_values(ascending=False)  # Making MEDV ie price column on top most positive correlation and checking all other columns correlation wrt MEDV columns

## here we are observing MEDV columns have High +ve correlation with RM and high negative correlation with LSTAT feature 

## plottting scatter plot to see the correlation effect of each attributes 

from pandas.plotting import scatter_matrix   # pandas plotting tool is utilized

attributes =['MEDV' , 'RM' , 'LSTAT']

scatter_matrix(housing[attributes] , figsize=(10,2))
plt.show()

housing.plot(kind = 'scatter' , x = 'RM' , y = 'MEDV' , alpha = 0.8)
plt.title('MEDV vs RM plot')
plt.show()   # below plot shows The RM and MEDV having highly postive correlation

## combining two feature to creat one more feature 

housing['TAXRM'] = housing['TAX']/housing['RM']   # creating new feature TAXRM and checking it added or not in our data set

housing.head()   # TAXRM column is added in our dataset

housing.plot(kind = 'scatter' , x = 'TAXRM' , y = 'MEDV' , alpha = 0.8)
plt.title('MEDV vs TAXRM plot')
plt.show()    # below plot shows The TAXRM and MEDV having highly postive correlation

## Now we tackling with null values of our data set

## option 1 - Removing training example from data set which having null value 

a = housing.dropna(subset='RM')     # This option is not acceptable becuase if dataset is very small then we cant remove our training example from our dataset
a

a.shape

## option 2 - removing attribute or feature from our dataset

b = housing.drop(["RM"] , axis=1)  # This option is not acceptable becuase if the feature is important than we cant remove that feature
b.shape

## option 3 - Replacing null value by meadian or mean value of perticular feature in dataset

median = housing['RM'].median()

median

c = housing.fillna(median) 

c.info()

## We are considering option 3 for tackling with null values, Creating a class which do imputer process , In which the null value of each dataset features will be replace with the respective median value of each attributes/features

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

housing_og = housing.drop(["TAXRM"] , axis=1)

imputer.fit(housing_og)

imputer.statistics_    # Median of each attribute of our datasets

## transforming our imputer method into our dataset

x = imputer.fit_transform(housing_og)   # Created variable which having fit and tranformation data of our old housing dataset

housing_tr = pd.DataFrame(x , columns=housing_og.columns)  ## New Dataframe is created by applying imputer class with stretergy Median

housing_tr.describe()  

## Creating pipeline for appying methods of Imputer class and standard scaler
Some Generalized Machine Learning algorithm works well only when they received fetaures in the form of equal scale
Hence before doing any process we have to apply fetaure scaling on the each features of our dataset for getting better inshighs from the algorithm

## joblib is a platfrom used to store pipeline of ML models

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

## pipeline always received input inthe form of list -
whatever the methods we have to apply in the pipleline we have to store it in the form of list which mentioned as follows

strt_train_set.shape

## seperating features and labels from our training datasets

housing_feature = strt_train_set.drop(['MEDV'] , axis=1)  # Created Features from complete datasets

housing_label = strt_train_set['MEDV'].copy()         # Created Label from complete datasets

housing_feature.shape

housing_label.shape

Creating pipeline with Standard Scaler as an input for preprocessing the input data 
and using Imputer class for filling null values of each attribute with its median values

my_pipline = Pipeline([('imputer',SimpleImputer(strategy='median')) , ('std_sclaer', StandardScaler())])

## Selection of model - Linear Regression selected for getting contineous Values of Prices of House accodring to feature provided

from sklearn.linear_model import LinearRegression

model_lin = LinearRegression()

Preparing data for testing our model, so we can see our model is doing correct preduction or not

some_data = housing_feature.iloc[:5]
some_label = housing_label.iloc[:5]

passing our feature data from pipeline

prepared_data = my_pipline.fit_transform(housing_feature)

Predicting the data and comparing it with our actual model

model_lin.fit(prepared_data,housing_label)

pred_y = model_lin.predict(prepared_data)

pred_y[:5]

housing_label[:]

## Passing our Feature from pipeline to manage null values and keeping scale of each attribute in equal Scale

housing_feature_tr = my_pipline.fit_transform(housing_feature)

model_lin.fit(housing_feature_tr,housing_label)

predicted_values = model_lin.predict(housing_feature_tr)

Checking the score of our model , by using RMSE method 

from sklearn.metrics import mean_squared_error

MSE_lin = mean_squared_error(predicted_values, housing_label)

MSE_lin

RMSE_lin = np.sqrt(MSE_lin)
RMSE_lin

## as we seen in above values of RMSE , We seen that model is giving some large value of RMSE and hence we have to change our model and again we have to find score of our model

from sklearn.tree import DecisionTreeRegressor

model_dec = DecisionTreeRegressor()

model_dec.fit(housing_feature_tr, housing_label)

Y_pred =model_dec.predict(housing_feature_tr)

MSE_Dec = mean_squared_error(Y_pred, housing_label)

MSE_Dec

## zero MSE means model overfits on the training datasets , Model is now passing through all the training dataset points and hence we are getting zero errors
Defination of error is the squared value of diffrence between Actual label value to the predicted label values

## Tackling with overfitting problem - Cross Validation Method 

from sklearn.model_selection import cross_val_score

score = cross_val_score(model_dec , housing_feature_tr , housing_label , cv= 10 , scoring='neg_mean_squared_error')

rmse_score_dtr = np.sqrt(-score)

rmse_score_dtr

## As we seen earlier MSE of Decision tree regressor is 0.0 , but after applying Cross validation process we are getting actual value of RMSE of Decision tree Regressor it means that , The problem of over fitting is overcome by using Cross validation method

## creating function which gives score , mean and Standard Deviation of each model

def print_score(score):
    print("Score Value :" , score)
    print("Mean of score :" , score.mean())
    print("Standard Deviation of Score :" , score.std())


print_score(rmse_score_dtr)

## Appying cross validation method on Linear Regressior 

score_lin = cross_val_score(model_lin , housing_feature_tr, housing_label,cv=10, scoring='neg_mean_squared_error')

rmse_score_lin = np.sqrt(-score_lin)

print_score(rmse_score_lin)

## Selecting Random forest regressor to check its score value

from sklearn.ensemble import RandomForestRegressor

model_ran = RandomForestRegressor()

score_ran = cross_val_score(model_ran , housing_feature_tr, housing_label,cv=10, scoring='neg_mean_squared_error')

rmse_score_ran = np.sqrt(-score_ran)

print_score(rmse_score_ran)

## We are finalizing Random Forest Regressor for Housing price prediction
And storing our model into joblib file
## Joblib is a platfrom which is utlized to store our model and it will provide light weight pipeline into the python

from joblib import dump, load

model_ran.fit(housing_feature_tr,housing_label)

dump(model_ran, 'Dragon_Real_joblib')

## testing our model by using testing Datasets 

strt_test_set   # Recalling our test datasets

x_test = strt_test_set.drop(["MEDV"] , axis = 1)

y_test = strt_test_set["MEDV"].copy()

x_test_prep = my_pipline.fit_transform(x_test)

x_test_prep

model_ran.fit(housing_feature_tr,housing_label)

y_predict = model_ran.predict(x_test_prep)

print(y_predict[:10],list(y_test[:10]))

final_score = cross_val_score(model_ran, x_test_prep, y_test , cv = 3 , scoring='neg_mean_squared_error')

print_score(np.sqrt(-final_score))

## here we concluded our Machine Leanring for Housing Price Prediction for Real Estate Company 
We can use this model into another file also by loading our joblib file into it

prepared_data[100]

housing_label[100]

## Thank you!!!


