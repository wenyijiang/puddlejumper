import xgboost as xgb

dtrain = xgb.DMatrix('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\train.csv')

param = {'max_depth': 10, 'eta': 0.3, 'silent': 0, 'objective': 'binary:logistic'}
param['nthread'] = 1
param['eval_metric'] = 'auc'

num_round = 10
bst = xgb.train(param, dtrain, num_round)
#bst.save_model('0001.model')

dtest = xgb.DMatrix('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\test.csv')
ypred = bst.predict(dtest)
ypred.to_csv('C:\\Users\\wq\\Desktop\\kaggle_credict_data\\data\\xgboost_baseline.csv', index = False)
