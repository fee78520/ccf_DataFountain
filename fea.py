import pandas as pd
from util import feat_nunique,feat_sum,feat_count,feat_max,feat_mean,feat_min,feat_median
import numpy as np
import time


agg = pd.read_csv('../data/train_agg.csv', sep='\t')
log = pd.read_csv('../data/train_log.csv', sep='\t', parse_dates=['OCC_TIM'])
flg = pd.read_csv('../data/train_flg.csv', sep='\t')


agg_test = pd.read_csv('../data/test_agg.csv', sep='\t')
log_test = pd.read_csv('../data/test_log.csv', sep='\t', parse_dates=['OCC_TIM'])

# sample = pd.read_csv('../data/submit_sample.csv',sep='\t')


log['EVT_LBL_1'] = log['EVT_LBL'].apply(lambda x:x.split('-')[0])
log['EVT_LBL_2'] = log['EVT_LBL'].apply(lambda x:x.split('-')[1])
log['EVT_LBL_3'] = log['EVT_LBL'].apply(lambda x:x.split('-')[1])

log_test['EVT_LBL_1'] = log_test['EVT_LBL'].apply(lambda x:x.split('-')[0])
log_test['EVT_LBL_2'] = log_test['EVT_LBL'].apply(lambda x:x.split('-')[1])
log_test['EVT_LBL_3'] = log_test['EVT_LBL'].apply(lambda x:x.split('-')[1])
log_test = log_test.sort_values(by=['USRID', 'OCC_TIM']).reset_index(drop=True)

log = log.sort_values(by=['USRID', 'OCC_TIM']).reset_index(drop=True)

train = log.drop_duplicates('USRID').reset_index(drop=True)[['USRID']]
train = pd.merge(train, agg, on='USRID', how='left')
train = train.sort_values(by=['USRID']).reset_index(drop=True)
train=feat_count(train,log,['USRID'],'EVT_LBL_1','count_1')
# train=feat_mean(train,log,['USRID'],'EVT_LBL_2','mean_2')
# train=feat_mean(train,log,['USRID'],'EVT_LBL_1','mean_1')



log['OCC_TIM'] = log['OCC_TIM'].apply(lambda x:x.day)
log = log.sort_values(['USRID','OCC_TIM'])
log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)
# print(log['next_time'])
log = log.groupby(['USRID'],as_index=False)['next_time'].agg({
    'next_time_mean':np.mean,
    'next_time_std':np.std,
    'next_time_min':np.min,
    'next_time_max':np.max
})
log = log[['USRID', 'next_time_mean', 'next_time_std', 'next_time_min', 'next_time_max']]


train = pd.merge(train, log, on='USRID', how='left')
train = pd.merge(train, flg, on='USRID', how='left')
train.to_csv(r'../input/train.csv', index=None)
print(train)



test = log_test.drop_duplicates('USRID').reset_index(drop=True)[['USRID']]
test = pd.merge(test, agg_test, on='USRID',how='left')
test = test.sort_values(by=['USRID']).reset_index(drop=True)
test=feat_count(test,log_test,['USRID'],'EVT_LBL_1','count_1')

log_test['OCC_TIM'] = log_test['OCC_TIM'].apply(lambda x:x.day)
log_test = log_test.sort_values(['USRID','OCC_TIM'])
log_test['next_time'] = log_test.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)
# print(log['next_time'])
log_test = log_test.groupby(['USRID'],as_index=False)['next_time'].agg({
    'next_time_mean':np.mean,
    'next_time_std':np.std,
    'next_time_min':np.min,
    'next_time_max':np.max
})
log_test = log_test[['USRID', 'next_time_mean', 'next_time_std', 'next_time_min', 'next_time_max']]

test = pd.merge(test, log_test, on='USRID', how='left')
# test = pd.merge(test, on='USRID', how='left')
test['FLAG'] = -1
test.to_csv(r'../input/test.csv', index=None)

print(test)













