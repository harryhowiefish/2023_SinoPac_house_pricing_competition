# SinoPac 2023

## 競賽基本資料
#### 競賽題目：永豐AI GO競賽-攻房戰
#### 隊伍名稱：待業中
#### 訓練集資料量：(11751, 22)
#### 公開測試集資料量：5876 筆
#### 測試集資料量：5875 筆
#### 名次：20/972
#### 成果：MAPE 7.507978
[比賽說明連結](https://tbrain.trendmicro.com.tw/Competitions/Details/30)

## Python Packages used
numpy, pandas, catboost, scikit-learn, kmodes, pyproj

## 技術說明
- data cleaning
    - remove outlier data
    - Adjust skewness of target price (Log transform)
    - Individually correct some inaccurate data from using information in the 備註 column
- feature engineering
    - include a lot of external data
    - kmean and kmode different features
    - create clusters from geographical data and perform aggregation.
    - include the external location data (schools, bus stop, mrt station...etc) with different ranging radius
- model design
    - param
        - learning_rate=0.01
        - depth=8
        - loss_function='RMSE'
        - class_weights=[1,220]
        - grow_policy = 'Depthwise'
        - l2_leaf_reg=2
        - random_strength=5
        - rsm=0.5
        - eval_metric = MAPE
- key concept
    - Adjust skewness of target (Log transform)
    - Create a lot of statistical features 
    - Feature selection (using both catboost provided method and feature importance)

## 使用資料：
- 主辦單位提供
    - training_data.csv
    - public_testing.csv
    - private_dataset.csv
    - external_data (包含13份csv檔)
- 外部資料
    - 108~111實價登錄資料（清洗後用Google Geocoding API定位）
    - 環境資料 open_data (空污)
    - 人口資料 open data (結婚資料,人口密度,出生資料,)
    - 用open data的「年齡分佈_109,年齡分佈_111」計算各年齡層人口變化


## 執行流程:
```
# 安裝所需套件
$ pip install -r requirements.txt 

# data_prep包含 Preprocessing, feature_engineering
$ python data_prep.py

#training and export result
$ python training_and_inference.py


```

## Action list
- add documentation
- create some pipeline framework
- attach a presentation file