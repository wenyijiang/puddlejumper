import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
#print(os.listdir("/host/home/kagglehomecredit/kaggledata/"))
app_train = pd.read_csv('/host/home/kagglehomecredit/kaggledata/application_train.csv')
#print('Training data shape: ', app_train.shape)
#app_train.head()
app_test = pd.read_csv('/host/home/kagglehomecredit/kaggledata/application_test.csv')
#print('Testing data shape: ', app_test.shape)
#app_test.head()
#print(app_train['TARGET'].value_counts())
#app_train['TARGET'].astype(int).plot.hist() #draw the picture
#plt.show() #show the picture
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
            columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
                '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
               "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns
missing_values = missing_values_table(app_train)
#print(missing_values.head(20))
#print(app_train.dtypes.value_counts())
#print(app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0))
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
#print('%d columns were label encoded.' % le_count)
##pd.get_dummies is one-hot encode
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)
#print('Training Features shape: ', app_train.shape)
#print('Testing Features shape: ', app_test.shape)
train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

#print('Training Features shape: ', app_train.shape)
#print('Testing Features shape: ', app_test.shape)

#print((app_train['DAYS_BIRTH'] / -365).describe())
#print(app_train['DAYS_EMPLOYED'].describe())
#app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
#plt.xlabel('Days Employment');
#plt.show()

anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
#print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
#print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
#print('There are %d anomalous days of employment' % len(anom))

# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

#app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
#plt.xlabel('Days Employment');
##plt.show()
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

##print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))
#correlations = app_train.corr()['TARGET'].sort_values()

# Display correlations
#print('Most Positive Correlations:\n', correlations.tail(15))
#print('\nMost Negative Correlations:\n', correlations.head(15))
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
##print(app_train['DAYS_BIRTH'].corr(app_train['TARGET']))
#plt.style.use('fivethirtyeight')

## Plot the distribution of ages in years
#plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
#plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
#plt.show()
#plt.figure(figsize = (10, 8))

## KDE plot of loans that were repaid on time
#sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

## KDE plot of loans which were not repaid on time
#sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

## Labeling of plot
#plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
#plt.show()
#age_data = app_train[['TARGET', 'DAYS_BIRTH']]
#age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

## Bin the age data
#age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
##print(age_data.head(10))
#age_groups  = age_data.groupby('YEARS_BINNED').mean()
##print(age_groups)


#plt.figure(figsize = (8, 8))

## Graph the age bins and the average of the target as a bar plot
#plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

## Plot labeling
#plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
#plt.title('Failure to Repay by Age Group');
#plt.show()
#ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
#ext_data_corrs = ext_data.corr()
##print(ext_data_corrs)
#plt.figure(figsize = (8, 6))

## Heatmap of correlations
#sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
#plt.title('Correlation Heatmap');
#plt.show()



#plt.figure(figsize = (10, 12))

## iterate through the sources
#for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    
    ## create a new subplot for each source
    #plt.subplot(3, 1, i + 1)
    ## plot repaid loans
    #sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    ## plot loans that were not repaid
    #sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    ## Label the plots
    #plt.title('Distribution of %s by Target Value' % source)
    #plt.xlabel('%s' % source); plt.ylabel('Density');
    
#plt.tight_layout(h_pad = 2.5)
    


#plt.show()

# Make a new dataframe for polynomial features
poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

# imputer is for handling missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns = ['TARGET'])

# Need to impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures

# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)
# Train the polynomial features
poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
#print('Polynomial Features shape: ', poly_features.shape)
#print(poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:34])

poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Add in the target
poly_features['TARGET'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Display most negative and most positive
#print(poly_corrs.head(10))
#print(poly_corrs.tail(5))

poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Merge polynomial features into training dataframe
poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

# Merge polnomial features into testing dataframe
poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Align the dataframes
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)

# Print out the new shapes
#print('Training data with polynomial features shape: ', app_train_poly.shape)
#print('Testing data with polynomial features shape:  ', app_test_poly.shape)


#train
from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()
    
# Feature names
features = list(train.columns)

# Copy of the testing data
test = app_test.copy()

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(app_test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# Train on the training data
log_reg.fit(train, train_labels)
log_reg_pred = log_reg.predict_proba(test)[:, 1]
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

print(submit.head())
submit.to_csv('log_reg_baseline.csv', index = False)