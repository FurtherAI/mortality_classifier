import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import pickle

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
    'n_estimators':[50, 200, 400, 750, 1000],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'max_leaves': [0, 1, 2, 3],  # no limit on leaves might be best
    'grow_policy': ['lossguide', 'depthwise'],  # favor splitting nodes with highest change in loss, alternatively, 'depthwise'
    'gamma': [0, 0.1, 1, 10],  # min loss reduction required for leaf split; higher = more conservative
    'learning_rate': [0.001, 0.01, 0.1, 0.2],  # default 0.3, common to be less than that
    'objective': ['binary:logistic', 'binary:hinge'],
    'tree_method': ['gpu_hist', 'hist', 'approx', 'auto', 'exact'],
    'subsample': [0.5, 0.6, 0.7],
    'sampling_method': ['uniform','gradient_based'],
    'alpha': [0, 1],
    'lambda': [0, 1],  # default for alpha and lambda (L1 and L2 regularization respectively)
    'scale_pos_weight': [1, (y == 0).sum() / (y == 1).sum()], # for imbalanced classes, can set to 1 for no balancing
    'max_bins': [128, 256, 512, 1024],  # number of bins used to discretize continuous variables (higher = more computation, maybe better accuracy)
    'validate_parameters': [True],
    'enable_categorical': [True],
    'use_label_encoder': [False],
}         

# --------------------------------------------------------------

# Instantiate classifier and perform repeated cross validation on random search of hyperparameter combinations for optimal set
# WARNING: LONG TRAINING TIME Approx 2.5 hrs with 8 cores CPU.
xgb = XGBClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# Already run, results in RandomSearchXGB txt file
# search = RandomizedSearchCV(xgb, params, n_iter = 500, scoring = 'roc_auc', n_jobs = -1, cv = cv, random_state = 42)
# search = GridSearchCV(xgb, params, scoring = 'roc_auc', n_jobs = -1, cv = cv, random_state = 42)
result = search.fit(x_train, y_train)

optimized_params = result.best_params_

print('Best Score: %s' % result.best_score_) 
print('Best Hyperparameters: %s' % result.best_params_)


opt_xgb = result.best_estimator_
pickle.dump(opt_xgb, open('xgb_classifier.sav', 'wb'))

# --------------------------------------------------------------

# Load the trained model
# opt_xgb = pickle.load(open(xgb_classifier.sav, 'rb'))
# result = opt_xgb.score(x_test, y_test) # optional

# Fit on whole training datset
# opt_xgb.fit(x_train, y_train, eval_metric = 'auc', eval_set=[(x_test, y_test)], early_stopping_rounds=40)
# print(opt_xgb.evals_result())
# pickle.dump(opt_xgb, open('opt_xgb_classifier.sav', 'wb'))

# --------------------------------------------------------------

# instantiate the classifier 
# xgb = XGBClassifier(**optimized_params)

# fit the classifier to the training data
# xgb.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=40) # number of rounds within which validation loss has to improve (rounds = new trees)

# print(xgb.evals_result())

# param_grid = {"max_depth":    [4, 5],
#               "n_estimators": [500, 600, 700],
#               "learning_rate": [0.01, 0.015]}

# # try out every combination of the above values
# search = GridSearchCV(xgb_rgr, param_grid, cv=5).fit(X_train, y_train)

# print("The best hyperparameters are ",search.best_params_)

