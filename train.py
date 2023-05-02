import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

params = {
    'n_estimators':400,
    'max_depth': 5,
    'max_leaves': 0,  # no limit on leaves
    'grow_policy': 'lossguide',  # favor splitting nodes with highest change in loss, alternatively, 'depthwise'
    'learning_rate': 0.1,  # default 0.3, common to be less than that
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',  # because I have gpu, may want to try exact because dataset is relatively small, or approx/hist if that is too slow
    'sampling_method': 'gradient_based',  # for gpu_hist only, can alternatively be uniform
    'alpha': 0,
    'lambda': 1,  # default for alpha and lambda (L1 and L2 regularization respectively)
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),  # for imbalanced classes, can set to 1 for no balancing
    'max_bins': 512,  # number of bins used to discretize continuous variables (higher = more computation, maybe better accuracy)
    'validate_parameters': True,
    'enable_categorical': True,
    'eval_metric': ['auc'],
    'use_label_encoder': False,
}         
           
# instantiate the classifier 
xgb = XGBClassifier(**params)

# fit the classifier to the training data
xgb.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=40) # number of rounds within which validation loss has to improve (rounds = new trees)

# print(xgb.evals_result())

# param_grid = {"max_depth":    [4, 5],
#               "n_estimators": [500, 600, 700],
#               "learning_rate": [0.01, 0.015]}

# # try out every combination of the above values
# search = GridSearchCV(xgb_rgr, param_grid, cv=5).fit(X_train, y_train)

# print("The best hyperparameters are ",search.best_params_)

