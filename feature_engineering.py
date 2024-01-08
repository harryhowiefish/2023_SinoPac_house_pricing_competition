import pandas as pd
import numpy as np
import catboost as cb
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from sklearn.cluster import KMeans
from kmodes import kmodes
import sys

def load_data(path:str) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    train = pd.read_csv(f'{path}/training_data.csv')
    test = pd.read_csv(f'{path}/public_testing.csv')
    pt = pd.read_csv(f'{path}/private_dataset.csv')
    test = pd.concat([test,pt])
    test.reset_index(drop=True,inplace=True)
    ext_trading = pd.read_csv(f'{path}/ext_trading_processed.csv')
    ext_trading = ext_trading.iloc[ext_trading[['單價元平方公尺','橫坐標','縱坐標']].dropna().index]
    ext_trading = ext_trading.query("單價元平方公尺>1e+04 & 單價元平方公尺<1e+06")
    ext_trading = ext_trading[ext_trading['建物型態']!='透天厝']
    ext_trading = ext_trading.reset_index(drop=True)

    return train,test,ext_trading

def drop_outlier(df:pd.DataFrame)->pd.DataFrame:
    df = df[df['主建物面積']<6]
    df = df[df['土地面積']<7]
    df = df[df['車位面積']<6]
    df = df[df['陽台面積']<6]
    df = df[df['附屬建物面積']<12]
    df = df.reset_index(drop=True)
    return df

def data_correction(df:pd.DataFrame,index:int,col_name:str,data:any)->pd.DataFrame:
    df.loc[index,col_name]=data
    return df

def zoning_correction(df:pd.DataFrame,reference_df:pd.DataFrame,r:int=100)->pd.DataFrame:
    for row in tqdm(df.iloc):
        data = reference_df[(reference_df['鄉鎮市區']==row['鄉鎮市區'])]
        data = data[(data['橫坐標'].between(row['橫坐標']-r,row['橫坐標']+r)) & (data['縱坐標'].between(row['縱坐標']-r,row['縱坐標']+r))]
        try:
            if data['都市土地使用分區'].value_counts().index[0] in ['住', '其他', '商', '農', '工']:
                df.loc[row.name,'使用分區']=data['都市土地使用分區'].value_counts().index[0]
        except:
            pass
    return df

def feat_engineering(table:pd.DataFrame)->pd.DataFrame:
    #top floor boolean
    table['top floor'] = table['移轉層次']==table['總樓層數']
    #New_town =city + town
    table['new_town'] = table['縣市']+table['鄉鎮市區']
    #建物與土地面積比例
    table['ratio'] = table['主建物面積']/table['土地面積']
    table['log_ratio'] = table['主建物面積'] - table['土地面積']
    #交易樓層高度比例
    table['floor_ratio'] = table['移轉層次']/table['總樓層數']
    road_name = table['路名'].to_list()
    for word in ['路','巷','街']:
        table[word]=[word in i for i in road_name]
    table['段']=[i[-1]=="段" in i for i in road_name]
    
    age_pct = []
    for row in table.iloc:
        if row['主要建材'] in ['加強磚造','磚造']:
            age_pct.append(row['屋齡']/35)
        else:
            age_pct.append(row['屋齡']/55)
    table['age_pct'] = age_pct
    return table

def location_k_cluster(train,test,ext_trading,cluster=500):
    all_data = pd.concat([train,test],axis=0)
    kmeans = KMeans(n_clusters=cluster,n_init='auto')
    kmeans.fit(train[['橫坐標','縱坐標']].to_numpy())

    train_data = train[['橫坐標','縱坐標']].to_numpy()
    train = pd.concat([train,pd.DataFrame(kmeans.transform(train_data),columns=[f"k{i}" for i in range(1,cluster+1)])],axis=1)
    test_data = test[['橫坐標','縱坐標']].to_numpy()
    test = pd.concat([test,pd.DataFrame(kmeans.transform(test_data),columns=[f"k{i}" for i in range(1,cluster+1)])],axis=1)

    train['cluster'] = kmeans.predict(train[['橫坐標','縱坐標']].to_numpy())
    test['cluster'] = kmeans.predict(test[['橫坐標','縱坐標']].to_numpy())
    all_data['cluster'] = kmeans.predict(all_data[['橫坐標','縱坐標']].to_numpy())

    ext_trading['cluster'] = kmeans.predict(ext_trading[['橫坐標','縱坐標']].to_numpy())

    for table in [train,test]:    
        map = all_data.groupby('cluster')['ID'].count().to_dict()
        table["cluster_based_internal_sales"]=table['cluster'].map(map)
        map = all_data.groupby('cluster')['主要用途'].value_counts(normalize=True).to_dict()
        table["cluster_based_purpose_ratio"]=table.apply(lambda x:map[(x['cluster'],x['主要用途'])],axis=1)
        map = all_data.groupby('cluster')['建物型態'].value_counts(normalize=True).to_dict()
        table["cluster_based_build_type_ratio"]=table.apply(lambda x:map[(x['cluster'],x['建物型態'])],axis=1)
        map = train.groupby('cluster')['單價'].mean().to_dict()
        table['cluster_internal_avgprice'] = table['cluster'].map(map)
    return train,test

def feature_k_cluster(train,test):
    all_data = pd.concat([train,test],axis=0)
    for name,cluster_num,cluster_col in zip(['car','size','floor'],[20,50,30],[['車位面積','車位個數'],['主建物面積','土地面積','陽台面積','附屬建物面積'],['移轉層次', '總樓層數']]):
        kmeans = KMeans(n_clusters=cluster_num,n_init='auto')
        kmeans.fit(all_data[cluster_col])

        train = pd.concat([train,pd.DataFrame(kmeans.transform(train[cluster_col].to_numpy()),
                                            columns=[f"{name}_k{i}" for i in range(1,cluster_num+1)])],axis=1)
        test = pd.concat([test,pd.DataFrame(kmeans.transform(test[cluster_col].to_numpy()),
                                            columns=[f"{name}_k{i}" for i in range(1,cluster_num+1)])],axis=1)
        train[f'{name}_cluster'] = kmeans.predict(train[cluster_col].to_numpy())
        test[f'{name}_cluster'] = kmeans.predict(test[cluster_col].to_numpy())
    return train,test



def k_mode(train,test,k_mode_col):
    km = kmodes.KModes(n_clusters=20, init='Huang', n_init=10, verbose=0)
    cat_data = pd.concat([train[k_mode_col],test[k_mode_col]],axis=0).values
    km.fit(cat_data)
    train['kmode_cluster'] = km.predict(train[k_mode_col])
    test['kmode_cluster'] = km.predict(test[k_mode_col])
    return train,test


def add_ext_trading(table,ext_trading):
    radius_options = [50,100,200,300,500,1000]
    map = ext_trading.groupby('cluster')['單價元平方公尺'].mean().to_dict()
    table['cluster_external_avgprice'] = table['cluster'].map(map)
    sale_count = []
    avg_price = []
    road_avg_price = []
    for idx,row in tqdm(enumerate(table.iloc)):
        x,y = row['橫坐標'],row['縱坐標']
        data = ext_trading[(ext_trading['土地位置建物門牌']==row['路名'])]
        road_avg_price.append(data['單價元平方公尺'].agg(['median']).values[0])
    
        temp = ext_trading[['橫坐標','縱坐標','單價元平方公尺']].copy()
        count_sub_list=[]
        price_sub_list=[]
        for r in radius_options[::-1]:
            temp = temp[(temp['橫坐標'].between(x-r,x+r)) & (temp['縱坐標'].between(y-r,y+r))]
            count_sub_list.append(temp.shape[0])
            price_sub_list.append(temp['單價元平方公尺'].agg('mean'))
        sale_count.append(count_sub_list)
        avg_price.append(price_sub_list)
    sale_count = np.array(sale_count).T
    avg_price = np.array(avg_price).T
    road_avg_price = np.array(road_avg_price).T
    table.loc[:,'external_road_median_price']=road_avg_price
    for idx,r in enumerate(radius_options[::-1]):
        table[f"ext_radius_{r}_sales"]=sale_count[idx]
        table[f"ext_radius_{r}_avgprice"]=avg_price[idx]
    return table

def add_ext_location(path,table):
    ext_loc = pd.read_csv(path)
    external_types = ext_loc.Name.unique()
    radius_options = [50,100,200,300,500,1000]
    for col in external_types:
        table.loc[:,f"dist_to_{col}"]=np.nan
        for r in radius_options[::-1]:
            table.loc[:,f"{col}_{r}"]=0
    for index,row in tqdm(enumerate(table[['橫坐標','縱坐標']].to_numpy())):
        for ext_type in external_types:
            table.loc[index,f"dist_to_{ext_type}"]=cdist([row],ext_loc[ext_loc['Name']==ext_type][['lng','lat']].to_numpy()).min()
        [x,y] = row
        temp = ext_loc.copy()
        for r in radius_options[::-1]:
            temp = temp[temp['lng'].between(x-r,x+r) & temp['lat'].between(y-r,y+r)]
            result = temp.Name.value_counts().to_dict()
            for key,value in result.items():
                table.loc[index,f"{key}_{r}"]=value
    return table

def add_open_data(folder_path,table):
    aqi = pd.read_csv(f'{folder_path}/aqi_processed.csv',index_col=0)
    marriage = pd.read_csv(f'{folder_path}/marriage.csv',index_col=0)
    birth = pd.read_csv(f'{folder_path}/birth.csv',index_col=0)
    birth = birth[['總計','avg']]

    for col in aqi.columns:
        table[col] = table['縣市'].map(aqi[col].to_dict())
    for col in marriage.columns:
        table[col] = table['縣市'].map(marriage[col].to_dict())
    for col in birth.columns:
        table[col] = table['縣市'].map(birth[col].to_dict())
    #this can be simplified further
    pop_density = pd.read_csv(f'{folder_path}/pop_density_processed.csv',index_col=0)
    result = []
    for row in table['new_town']:
        if row in pop_density.index:
            result.append(pop_density.loc[row].to_numpy())
        else:
            if row[:3] in pop_density.index:  ###新竹市新竹市問題修正
                result.append(pop_density.loc[row[:3]].to_numpy())
            else:
                result.append(np.array([np.nan,np.nan,np.nan]))
    result = np.array(result).T
    for name,data in zip(pop_density.columns,result):
        table[f"town_{name}"]=data

    result = []
    for row in table['縣市']:
        if row[:3] in pop_density.index:  ###新竹市新竹市問題修正
            result.append(pop_density.loc[row[:3]].to_numpy())
        else:
            result.append(np.array([np.nan,np.nan,np.nan]))
    result = np.array(result).T
    for name,data in zip(pop_density.columns,result):
        table[f"city_{name}"]=data

    pop_change = pd.read_csv(f'{folder_path}/pop_change.csv',index_col=0)
    result = []
    for row in table['new_town']:
        if row in pop_change.index:
            result.append(pop_change.loc[row].to_numpy())
        else:
            if row[:3] in pop_change.index:  ###新竹市新竹市問題修正
                result.append(pop_change.loc[row[:3]].to_numpy())
            else:
                print(row)
                print('failed')
                result.append(np.array([np.nan,np.nan,np.nan]))
    result = np.array(result).T
    for name,data in zip(pop_change.columns,result):
        table[f"town_{name}"]=data
    result = []
    for row in table['縣市']:
        if row[:3] in pop_change.index:  ###新竹市新竹市問題修正
            result.append(pop_change.loc[row[:3]].to_numpy())
        else:
            print('failed')
            result.append(np.array([np.nan,np.nan,np.nan]))
    result = np.array(result).T
    for name,data in zip(pop_change.columns,result):
        table[f"city_{name}"]=data
    return table

def perfect_matching(table,ext_trading,dist=100):
    for row in tqdm(table.iloc):
        data = ext_trading[(ext_trading['土地位置建物門牌']==row['路名'])]
        data = data[(data['鄉鎮市區']==row['鄉鎮市區'])]
        # data = data[(data['總樓層數']==row['總樓層數']) &(data['移轉層次']==row['移轉層次'])]
        data = data[(data['建物型態']==row['建物型態']) & (data['主要建材']==row['主要建材'])]
        data = data[(data['總樓層數']==row['總樓層數']) &(data['移轉層次'].between(row['移轉層次']-2,row['移轉層次']+2))]
        data['dist'] = cdist(data[['橫坐標','縱坐標']].to_numpy(),[row[['橫坐標','縱坐標']].to_list()])
        data = data[data['dist']<dist]
        if len(data)>=1:
            temp = pd.DataFrame(row).transpose()
            temp['true_建物移轉總面積'] = data['建物移轉總面積平方公尺'].median()
            temp['true_單價'] = data['單價元平方公尺'].median()
            table.loc[row.name,'true_單價']=data['單價元平方公尺'].median()
            table.loc[row.name,'true_建物移轉總面積']=data['建物移轉總面積平方公尺'].median()
    return table

def external_data_target_encoding(train,test,ext_trading,cluster=50):
    kmeans = KMeans(n_clusters=cluster,n_init='auto')
    kmeans.fit(train[['橫坐標','縱坐標']].to_numpy())
    train['cluster_50'] = kmeans.predict(train[['橫坐標','縱坐標']].to_numpy())
    test['cluster_50'] = kmeans.predict(test[['橫坐標','縱坐標']].to_numpy())

    ext_trading['cluster_50'] = kmeans.predict(ext_trading[['橫坐標','縱坐標']].to_numpy())

    #External trading info
    #neighbor area sales count and median_price
    radius_options = [200,1000,2000]
    median_map = ext_trading.groupby('cluster_50')['單價元平方公尺'].median().to_dict()
    min_map = ext_trading.groupby('cluster_50')['單價元平方公尺'].min().to_dict()

    for table in [train,test]:
        table['cluster_50_external_median_price'] = table['cluster_50'].map(median_map)
        table['cluster_50_external_minprice'] = table['cluster_50'].map(min_map)
        median_price = []
        road_med_price = []
        for idx,row in tqdm(enumerate(table.iloc)):
            x,y = row['橫坐標'],row['縱坐標']
            data = ext_trading[(ext_trading['土地位置建物門牌']==row['路名'])]
            road_med_price.append(data['單價元平方公尺'].median())
            
            temp = ext_trading[(ext_trading['建物型態']==row['建物型態']) & (ext_trading['鄉鎮市區']==row['鄉鎮市區'])]
            price_sub_list=[]
            for r in radius_options[::-1]:
                temp = temp[(temp['橫坐標'].between(x-r,x+r)) & (temp['縱坐標'].between(y-r,y+r))]
                price_sub_list.append(temp['單價元平方公尺'].median())
            median_price.append(price_sub_list)
        median_price = np.array(median_price).T
        road_med_price = np.array(road_med_price).T
        table.loc[:,'external_road_median_price']=road_med_price
        for idx,r in enumerate(radius_options[::-1]):
            table[f"ext_radius_{r}_median_price"]=median_price[idx]
    return train,test
def make_all_feature(path:str)->None:

    train,test,ext_trading = load_data(path)
    train = drop_outlier(train)
    train = data_correction(train,781,'主要建材','鋼骨鋼筋混凝土造')
    train = data_correction(train,4750,'主要建材','鋼骨鋼筋混凝土造')
    train = data_correction(train,5608,'主要建材','鋼骨鋼筋混凝土造')
    train = data_correction(train,5639,'主要建材','鋼骨鋼筋混凝土造')
    train = data_correction(train,8057,'主要建材','鋼骨鋼筋混凝土造')
    for table in [train,test]:
        table = zoning_correction(table,ext_trading)
        table = feat_engineering(table)
    
    train,test = location_k_cluster(train,test,ext_trading)
    train,test = feature_k_cluster(train,test)
    train,test = k_mode(train,test,k_mode_col=['主要建材','建物型態'])
    for table in [train,test]:
        table = add_ext_trading(table,ext_trading)
        table = add_ext_location(f'{path}/external_data/external_transformed_simplified.csv',table)
        table = add_open_data(f'{path}/open_data',table)
        table = perfect_matching(table,ext_trading)
    
    train,test = external_data_target_encoding(train,test,ext_trading)
    print(train.shape,test.shape)
    return train,test

if __name__ == "__main__":
    if len(sys.argv)<2:
        print('missing data path')
        sys.exit()
    path = sys.argv[1]
    train,test = make_all_feature(path)
    train.to_csv('final_train_processed.csv',index=False)
    test.to_csv('final_test_processed.csv',index=False)