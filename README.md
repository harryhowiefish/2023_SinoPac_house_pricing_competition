# SinoPac 2023

## 隊伍名稱：待業中

## 作者：harryhowiefish

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
