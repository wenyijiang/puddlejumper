import xgboost as xgb
import time
from sklearn import cross_validation
import numpy as np
import pandas as pd

# dtrain = xgb.DMatrix('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\train.csv')
#
# param = {'max_depth': 10, 'eta': 0.3, 'silent': 0, 'objective': 'binary:logistic'}
# param['nthread'] = 1
# param['eval_metric'] = 'auc'
#
# num_round = 10
# bst = xgb.train(param, dtrain, num_round)
# #bst.save_model('0001.model')
#
# dtest = xgb.DMatrix('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\test.csv')
# ypred = bst.predict(dtest)
# ypred.to_csv('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\xgboost_baseline.csv', index = False)

start_time = time.time()
train = np.array(pd.read_csv('/host/home/kagglehomecredit/kaggledata/train.csv').values.tolist())
test = np.array(pd.read_csv('/host/home/kagglehomecredit/kaggledata/test.csv').values.tolist())

train_label = train[0:1]
train_data = train[1:len(train)]

print(len(train_data))
print(len(test))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data, train_label, test_size=0.33, random_state=42)

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test = xgb.DMatrix(X_test, label=y_test)
##参数
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    # 'nthread':7,# cpu 线程数 默认最大
    'eta': 0.007,  # 如同学习率
    'min_child_weight': 3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'max_depth': 15,  # 构建树的深度，越大越容易过拟合
    'gamma': 0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    # 'alpha':0, # L1 正则项参数
    'scale_pos_weight': 1,  # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
    # 'objective': 'multi:softmax', #多分类的问题
    # 'num_class':10, # 类别数，多分类与 multisoftmax 并用
    'seed': 1000,  # 随机种子
    'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 100  # 迭代次数
watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]

# 训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
# model.save_model('./model/xgb.model') # 用于存储训练出的模型
print("best best_ntree_limit", model.best_ntree_limit)
y_pred = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
# 输出运行时长
cost_time = time.time() - start_time
print("xgboost success!", '\n', "cost time:", cost_time, "(s)......")

# param = {'max_depth': 15, 'eta': 0.01, 'silent': 0, 'objective': 'binary:logistic', 'scale_pos_weight':'10'}
# param['nthread'] = 1
# param['eval_metric'] = 'auc'
#
# num_round = 20
# bst = xgb.train(param, dtrain, num_round)
# bst.save_model('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\xgboost.0001.model')
#
test_id = test[0:1]

dtest = xgb.DMatrix(test)
ypred = model.predict(dtest)
result = open('/host/home/kagglehomecredit/kaggledata/xgboost_baseline.csv', 'w')
result.write("SK_ID_CURR,TARGET\n")
ypred = ypred.tolist()
for i in range(0, len(ypred)):
    result.write(str(test_id[i]) + "," + str(ypred[i]))
    result.write("\n")
result.close()
