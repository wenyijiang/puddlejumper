#numpy and pandas for data manipulation

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

#matplotlib and seaborn for plotting

print(os.listdir("C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data"))

app_train = pd.read_csv('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\application_train.csv')
app_test = pd.read_csv('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\application_test.csv')
# print('Training data shape: ', app_train.shape)
#
# print(app_train['TARGET'].value_counts())
# app_train['TARGET'].astype(int).plot.hist()
#plt.show()

def missing_values_table(df):
    #Total missing values
    mis_val = df.isnull().sum()

    #Percentage of missing values
    mis_val_percent = 100 * mis_val / len(df)

    #Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)

    # Rename the colums
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '%of Total Values'})

    #Sort the talble bny percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('%of Total Values', ascending = False).round(1)

    #Print some summary information
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

#Iterate through the columns
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
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

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
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
# plt.xlabel('Days Employment');
#plt.show()

app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

# Find correlations with the target and sort
# correlations = app_train.corr()['TARGET'].sort_values()
#
# # Display correlations
# print('Most Positive Correlations:\n', correlations.tail(15))
# print('\nMost Negative Correlations:\n', correlations.head(15))

# Find the correlation of the positive days since birth and target
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
#print(app_train['DAYS_BIRTH'].corr(app_train['TARGET']))

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
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
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


def mkString(array):
    s = str(array(0))
    for i in range(1, len(array)):
        s = s + "," + str(array(i))
    return s



# f1 = open('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\train.csv', 'w')
#
# label = train_labels.tolist()
#
# for i in range(0, len(train)):
#     f1.write(str(label[i]) + "," + mkString(train(i)))
#     f1.write("\n")
# f1.close()
#
#
# f2 = open('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\test.csv', 'w')
#
# for array in test:
#     f2.write(mkString(array) + "\n")
# f2.close()
# from sklearn.ensemble import RandomForestClassifier
#
# # Make the random forest classifier
# random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
#
# # Train on the training data
# random_forest.fit(train, train_labels)
#
# # Extract feature importances
# feature_importance_values = random_forest.feature_importances_
# feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
#
# # Make predictions on the test data
# predictions = random_forest.predict_proba(test)[:, 1]
#
# # Make a submission dataframe
# submit = app_test[['SK_ID_CURR']]
# submit['TARGET'] = predictions
#
# # Save the submission dataframe
# submit.to_csv('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\random_forest_baseline.csv', index = False)

import xgboost as xgb
from sklearn import cross_validation
import time
start_time = time.time()
dtrain = xgb.DMatrix(train, label = train_labels)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, train_labels, test_size=0.001, random_state=42)


xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test = xgb.DMatrix(X_test,label=y_test)
##参数
params={
'booster':'gbtree',
'objective': 'binary:logistic',
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
#'nthread':7,# cpu 线程数 默认最大
'eta': 0.007, # 如同学习率
'min_child_weight':3,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'max_depth':10, # 构建树的深度，越大越容易过拟合
'gamma':0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#'alpha':0, # L1 正则项参数
'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
#'objective': 'multi:softmax', #多分类的问题
#'num_class':10, # 类别数，多分类与 multisoftmax 并用
'seed':1000, #随机种子
'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 100 # 迭代次数
watchlist = [(xgb_train, 'train'),(xgb_test, 'val')]

#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
#model.save_model('./model/xgb.model') # 用于存储训练出的模型
print("best best_ntree_limit",model.best_ntree_limit)
y_pred = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
#输出运行时长
cost_time = time.time()-start_time
print( "xgboost success!",'\n',"cost time:",cost_time,"(s)......")



# param = {'max_depth': 15, 'eta': 0.01, 'silent': 0, 'objective': 'binary:logistic', 'scale_pos_weight':'10'}
# param['nthread'] = 1
# param['eval_metric'] = 'auc'
#
# num_round = 20
# bst = xgb.train(param, dtrain, num_round)
# bst.save_model('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\xgboost.0001.model')
#
test_id = app_test['SK_ID_CURR']

dtest = xgb.DMatrix(test)
ypred = model.predict(dtest)
result = open('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\xgboost_baseline.csv', 'w')
result.write("SK_ID_CURR,TARGET\n")
ypred = ypred.tolist()
for i in range(0, len(ypred)):
    result.write(str(test_id[i]) + "," + str(ypred[i]))
    result.write("\n")
result.close()



