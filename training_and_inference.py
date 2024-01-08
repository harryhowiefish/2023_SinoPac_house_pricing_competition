import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.cluster import KMeans
from kmodes import kmodes

CAT_COL = ['縣市','new_town','鄉鎮市區','使用分區','主要建材','建物型態','主要用途']

class mapeMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * abs((np.e**approx[i] - np.e**target[i])/np.e**target[i])

        return error_sum, weight_sum

def training(model:cb.core.CatBoostRegressor,path:str):
    train = pd.read_csv('final_train_processed.csv')
    for col in CAT_COL:
        train[col].fillna('',inplace=True)
    target = train['單價']
    train.drop(['ID','路名', '橫坐標', '縱坐標', '備註'],axis=1,inplace=True)
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=0)
    train_dataset = cb.Pool(X_train, np.log(y_train),cat_features=CAT_COL) 
    val_dataset = cb.Pool(X_val, np.log(y_val),cat_features=CAT_COL) 
    model.select_features(train_dataset,eval_set=val_dataset,features_for_select=train.columns,num_features_to_select=422)

def predict(model:cb.core.CatBoostRegressor,path:str):
    test = pd.read_csv(path)
    for col in CAT_COL:
        test[col].fillna('',inplace=True)
    test.drop(['ID'],axis=1,inplace=True)
    inference = cb.Pool(test,cat_features=CAT_COL) 
    pred = model.predict(inference)
    pred = np.e**pred
    sub = pd.read_csv('./data/public_private_submission_template.csv')
    sub['predicted_price'] = pred
    sub.to_csv('./submission.csv',index=False)

def main():
    model = cb.CatBoostRegressor(loss_function="RMSE",
                            od_type='Iter', od_wait=2000,
                            verbose=300,depth=8,iterations=8300,
                            grow_policy='Depthwise',
                            rsm=0.5,learning_rate=0.01,l2_leaf_reg=2,random_strength=5,eval_metric=mapeMetric(),random_seed=5)
    model = training(model,'./final_train_processed.csv')
    model = predict(model,'final_test_processed.csv')

if __name__ == "__main__":
    main()