import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import pickle
import json

x = pd.read_csv('processed_train_x_v2.csv')
y = pd.read_csv('train_y.csv')

x.drop(columns=['Unnamed: 0', 'patientunitstayid'], inplace=True)
y.drop(columns=['Unnamed: 0', 'patientunitstayid'], inplace=True)  # already aligned from preprocessing
y = y.to_numpy().reshape(-1)

categorical_cols = ['cellattributevalue', 'ethnicity', 'gender']
for col in categorical_cols:
    x[col] = x[col].astype('category')

# print((y == 1).sum())  # 168
# print((y== 0).sum())  # 1848

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

params = {
    'verbosity': [0],
    'n_estimators':[60, 250, 400, 600],  # 50, 400, 750, 1000
    'max_depth': [3, 4, 6], # 1, 2, 3, 4, 5, 6, 7, 8
    'max_leaves': [1, 3, 5],  # 0, 2, 3 no limit on leaves might be best
    'grow_policy': ['lossguide', 'depthwise'],  # favor splitting nodes with highest change in loss, alternatively, 'depthwise'
    'gamma': [0, 1],  # 0.1, 1, 10 min loss reduction required for leaf split; higher = more conservative
    'learning_rate': [0.1, 0.2],  # 0.01, 0.001, 0.1, 0.2 default 0.3, common to be less than that
    'objective': ['binary:logistic', 'binary:hinge'],
    'tree_method': ['gpu_hist'], # , 'hist', 'approx', 'auto', 'exact'
    'subsample': [0.7], # 0.6, 0.5, 
    'sampling_method': ['uniform','gradient_based'],
    'alpha': [0, 1],
    'lambda': [0, 1],  # default for alpha and lambda (L1 and L2 regularization respectively)
    'scale_pos_weight': [1, (y == 0).sum() / (y == 1).sum()], # for imbalanced classes, can set to 1 for no balancing
    'max_bins': [256],  # 128, 512, 1024 number of bins used to discretize continuous variables (higher = more computation, maybe better accuracy)
    'validate_parameters': [True],
    'enable_categorical': [True],
    'use_label_encoder': [False],
}         

# --------------------------------------------------------------

# Instantiate classifier and perform repeated cross validation on random search of hyperparameter combinations for optimal set
# WARNING: LONG TRAINING TIME Approx 2.5 hrs with 8 cores CPU.
# xgb = XGBClassifier()
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# Already run, results in RandomSearchXGB txt file
# search = RandomizedSearchCV(xgb, params, n_iter = 500, scoring = 'roc_auc', n_jobs = -1, cv = cv, random_state = 42)
# search = GridSearchCV(xgb, params, scoring = 'roc_auc', n_jobs = -1, cv = cv, verbose = 2)
# result = search.fit(x_train, y_train)

# optimized_params = result.best_params_
# json_opt_params = json.dumps(optimized_params)
# with open('opt_params_dict.json', 'w') as out:
#     out.write(json_opt_params)


# print('Best Score: %s' % result.best_score_) 
# print('Best Hyperparameters: %s' % result.best_params_)


# opt_xgb = result.best_estimator_
# pickle.dump(opt_xgb, open('xgb_classifier.sav', 'wb'))

# --------------------------------------------------------------

# Load the trained model
# opt_xgb = pickle.load(open(xgb_classifier.sav, 'rb'))
# result = opt_xgb.score(x_test, y_test) # optional


## TRAIN THE MODEL WITH OPTIMAL PARAMETERS FOUND BY RANDOM SEARCH ON WHOLE DATASET
# Fit on whole training datset


opt_params = {
    'verbosity': 0, 
    'validate_parameters': True, 
    'use_label_encoder': False, 
    'tree_method': 'gpu_hist', 
    'subsample': 0.7, 
    'scale_pos_weight': 1, 
    'sampling_method': 'gradient_based',
    'objective': 'binary:logistic', 
    'n_estimators': 180, # reduced from 400 to 180 from random search guess
    'max_leaves': 3, 
    'max_depth': 4, 
    'max_bins': 256, 
    'learning_rate': 0.1, 
    'lambda': 0, 
    'grow_policy': 'lossguide', 
    'gamma': 1, 
    'enable_categorical': True, 
    'alpha': 1
}


opt_xgb = XGBClassifier(**opt_params)
opt_xgb.fit(x, y, eval_metric = 'auc', eval_set=[(x_test, y_test)], early_stopping_rounds=40)  # eval just to make sure it is training well

x_test = pd.read_csv('processed_test_x_v2.csv')
x_test_patient_ids = x_test['patientunitstayid']

x_test.drop(columns=['Unnamed: 0', 'patientunitstayid'], inplace=True)

categorical_cols = ['cellattributevalue', 'ethnicity', 'gender']
for col in categorical_cols:
    x_test[col] = x_test[col].astype('category')
    
pickle.dump(opt_xgb, open('opt_xgb_classifier.sav', 'wb'))

predictions = opt_xgb.predict_proba(x_test)[:, 1]
submission_df = pd.DataFrame(columns=['patientunitstayid', 'hospitaldischargestatus'])
submission_df['patientunitstayid'] = x_test_patient_ids.astype(np.int32)
submission_df['hospitaldischargestatus'] = predictions

submission_df.to_csv('submission.csv', index=False)
# --------------------------------------------------------------

