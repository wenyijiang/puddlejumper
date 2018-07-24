# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
# Suppress warnings
# File System manangement
import os
# Suppress warnings
import warnings

import numpy as np
import pandas as pd
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting

#print(os.listdir("/host/home/kagglehomecredit/kaggledata"))
# train_path = '/host/home/kagglehomecredit/kaggledata/application_train.csv'
# test_path = '/host/home/kagglehomecredit/kaggledata/application_test.csv'
# result_path = '/host/home/kagglehomecredit/kaggledata/lightgbm_baseline.csv'

train_path = 'C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\application_train.csv'
test_path = 'C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\application_test.csv'
result_path = 'C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\lightgbm_baseline.csv'

app_train = pd.read_csv(train_path)
app_test = pd.read_csv(test_path)


# print('Training data shape: ', app_train.shape)
#
# print(app_train['TARGET'].value_counts())
# app_train['TARGET'].astype(int).plot.hist()
# plt.show()

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * mis_val / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the colums
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '%of Total Values'})

    # Sort the talble bny percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '%of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# missing_values = missing_values_table(app_train)
# print(missing_values.head(20))
#
# print(app_train.dtypes.value_counts())
# print(app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0))

# Create a label encoder object
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

print('%d columns were label encoded.' % le_count)

# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join='inner', axis=1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

# print((app_train['DAYS_BIRTH'] / -365).describe())
#
# print(app_train['DAYS_EMPLOYED'].describe())

# app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
# plt.xlabel('Days Employment')
# plt.show()

# for attribute in app_train:
#     print("===================================")
#     print("attribute: " + attribute)
#     print(app_train[attribute].describe())
#     print("===================================")
# print(app_train[''])
# app_train['DAYS_EMPLOYED'].plot.hist(title='DAYS_EMPLOYED')
# plt.xlabel('DAYS_EMPLOYED')
# plt.show()

# anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
# non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
# print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
# print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
# print('There are %d anomalous days of employment' % len(anom))

# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

# app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
# plt.xlabel('Days Employment');
# plt.show()

app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

print(
    'There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

# Find correlations with the target and sort
# correlations = app_train.corr()['TARGET'].sort_values()
#
# # Display correlations
# print('Most Positive Correlations:\n', correlations.tail(15))
# print('\nMost Negative Correlations:\n', correlations.head(15))

# Find the correlation of the positive days since birth and target
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
# print(app_train['DAYS_BIRTH'].corr(app_train['TARGET']))

# Set the style of plots
# plt.style.use('fivethirtyeight')
#
# # Plot the distribution of ages in years
# plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
# plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
# plt.show()
#
# plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
# sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')
#
# # KDE plot of loans which were not repaid on time
# sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
#
# # Labeling of plot
# plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
# plt.show()

# Age information into a separate dataframe
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20, 70, num=11))
age_data.head(10)

app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']

from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns=['TARGET'])
else:
    train = app_train.copy()

# Feature names
features = list(train.columns)

# Copy of the testing data
test = app_test.copy()

# Median imputation of missing values
imputer = Imputer(strategy='median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

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




import json
import time
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
start_time = time.time()

X_train,X_test,y_train,y_test =train_test_split(train, train_labels,test_size=0.2)
# 加载你的数据
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'binary', # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 100,   # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1, # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    'is_unbalance':'true',
    'num_tree':500

}
print('Start training...')
# 训练 cv and train
gbm = lgb.train(params,lgb_train,num_boost_round=200,valid_sets=lgb_eval,early_stopping_rounds=100)
print('Save model...')
# 保存模型到文件
gbm.save_model('model.txt')

print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# 评估模型
cost_time = time.time() - start_time
print("lightgbm success!", '\n', "cost time:", cost_time, "(s)......")
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

test_id = app_test['SK_ID_CURR']
#test = lgb.Dataset(test)
y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)
result = open(result_path, 'w')
result.write("SK_ID_CURR,TARGET\n")
y_pred = y_pred.tolist()
for i in range(0, len(y_pred)):
    result.write(str(test_id[i]) + "," + str(y_pred[i]))
    result.write("\n")
result.close()


test_id = app_test['SK_ID_CURR']
#test = lgb.Dataset(test)
y_pred = gbm.predict(test)
result = open(result_path, 'w')
result.write("SK_ID_CURR,TARGET\n")
y_pred = y_pred.tolist()
for i in range(0, len(y_pred)):
    result.write(str(test_id[i]) + "," + str(y_pred[i]))
    result.write("\n")
result.close()