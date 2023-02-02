from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np 
import catboost as cb
from sklearn.multioutput import RegressorChain
import os
import gc 

#---------------------------------------------------------------------------------------
########################################################################
#---------------------------------------------------------------------------------------
file_name="output"
targets=["Available","Charging"	,"Passive","Other"]
def mkdir(d):
    """
        This function will create a folder.
        Parameters
        ----------
        d : string 
            This is the name your folder
        Returns
        -------
        No thing
    """
    if not os.path.exists(d):
        os.makedirs(d)
        print(f"Creating folder called : '{file_name}.'")
mkdir(file_name)
#---------------------------------------------------------------------------------------
########################################################################
def make_cols(data,labels=True):
    """
        This function will create new columns for time serais data"
        Parameters
        ----------
        data : DataFrame 
            A data Frame to extract time interval from column "date"
        labels : bool
            To create labale by summing targets or not.
        Returns
        -------
        data : DataFrame
            a data frame with new columns that has been created
    """
    if labels :
        data["labels"]=data.Available.astype(int).astype(str) + data.Charging.astype(int).astype(str) + data.Passive.astype(int).astype(str)+ data.Other.astype(int).astype(str)
    data['hour'] = data.date.dt.hour
    data['min'] = data.date.dt.minute
    data['quarter'] = data.date.dt.quarter
    data['dayofmonth'] = data.date.dt.day
    return data

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
############################        Stations        ############################
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#train set
print("We will start importing train set and test set for Station.")
train_station = pd.read_csv("data_to_use/train_onlyNext4EWM4andBack_EWM_remChar.csv", sep=",")
train_station['date'] = pd.to_datetime(train_station['date'])
train_station['Postcode'] = train_station['Postcode'].astype(str)
# test set
test_station=pd.read_csv("data_to_use/test.csv", sep=",")
test_station['date'] = pd.to_datetime(test_station['date'])
test_station['Postcode'] = test_station['Postcode'].astype(str)
#create cols
train_station=make_cols(train_station)
test_station=make_cols(test_station,False)
#choose the variables
col_act_station=["Station","area","hour","tod","dow","trend"]
X_train_station=train_station.copy()[col_act_station]
y_train_station=train_station.labels.copy()
cat_index_station = [0,1]
########################################################################
#model
print("Starting modelisation for Station... \n(It would take 2 min if you have a GPU otherwise it will take 15-20 min. Sorry !)")
clf = cb.CatBoostClassifier(#task_type="GPU",
                    eval_metric="TotalF1",one_hot_max_size=250,loss_function="MultiClass"
                            ,depth=4,iterations= 1200, l2_leaf_reg= 1, learning_rate= 0.1)
clf.fit(X_train_station,y_train_station, cat_features= cat_index_station,verbose=0)
print("Done modeling, now we are going forecast.")
########################################################################
#forcast
test_stationX=test_station[col_act_station].copy()
test_pred_class=clf.predict(test_stationX).ravel()
nm_lab=np.unique(train_station["labels"])
# retransform labels to 4 columns
dict_ofdata={key: [int(value[0]),int(value[1]),int(value[2]),int(value[3])] for key,value in zip(nm_lab,nm_lab) }
df44=test_station.copy()
df44[targets]=pd.DataFrame([dict_ofdata[i] for i in test_pred_class]).values
df44[['date', 'area', 'Station'] +
                  targets].to_csv(file_name+"/station.csv", index=False)
print("We are done forecasting Stations. a file station.csv is been saved in output folder.\n\n")
#Clear Memory
del train_station, X_train_station, y_train_station, test_stationX,df44,test_pred_class
gc.collect()

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
############################        Area        ############################
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

print("We will start importing train set and test set for Area")
train_station_A = pd.read_csv("data_to_use/data_ewm1.2.csv", sep=",")
train_station_A['date'] = pd.to_datetime(train_station_A['date'])
train_station_A['Postcode'] = train_station_A['Postcode'].astype(str)
state="Charging"
df7=train_station_A[["date" ,"Station",state]].pivot(index='date', columns='Station', values=state).copy()
index_rem=df7[df7.eq(df7.iloc[:, 5], axis=0).all(1)].index
#remove all values are equale
train_station_A=train_station_A[train_station_A.date.isin(index_rem)==False].copy()
#remove values is bigger or smaller than 3
train_station_A=train_station_A[((train_station_A[targets].sum(axis=1)>3)|(
     train_station_A[targets].sum(axis=1)<3))==False].reset_index(drop=1).copy()
#make col
train_station_A=make_cols(train_station_A,False)
train_area = train_station_A.groupby(['date', 'area']).agg({'Available': 'sum',
                                                         'Charging': 'sum',
                                                          'Passive': 'sum',
                                                          'Other': 'sum',
                                                          'tod': 'max',
                                                          'dow': 'max',
                                                          'Latitude': 'mean',
                                                          'Longitude': 'mean',
                                                          'trend': 'max',
                                                           'hour': 'max',
                                                          'min': 'max',
                                                           'quarter': 'max',
                                                          'dayofmonth': 'max',}).reset_index()
test_area = test_station.groupby(['date', 'area']).agg({
                                                          'tod': 'max',
                                                          'dow': 'max',
                                                          'Latitude': 'mean',
                                                          'Longitude': 'mean',
                                                          'trend': 'max',
                                                            'hour': 'max',
                                                          'min': 'max',
                                                           'quarter': 'max',
                                                          'dayofmonth': 'max',}).reset_index()
col_act_area=["min","trend"]
catt_col=["hour","tod","dow"]
enc = OneHotEncoder(handle_unknown='ignore')
X=train_area.copy()
enc.fit(X[catt_col])
nameHot=enc.get_feature_names_out()
X.loc[:,nameHot]=enc.transform(X[catt_col]).toarray()
########################################################
#model
print("Starting modelisation for Area... ")
models={}
# model by area
for ar in train_area.area.unique():
    area_x_train= X.loc[X.area==ar,col_act_area+list(nameHot)].copy()
    area_y_train=X.loc[X.area==ar,targets].copy()
    clf_area = cb.CatBoostRegressor(eval_metric="MultiRMSE",loss_function="MultiRMSE",one_hot_max_size=200,
                            depth= 3,iterations =1000, learning_rate= .1,l2_leaf_reg= 1)# 
    chain=RegressorChain(clf_area,order=[0,1,2,3]).fit(area_x_train,area_y_train, cat_features= 
                                                       [],verbose=0)
    models[ar]=chain
print("Done modeling, now we are going to forecast.")
########################################################
#output
allfortest=[]
df_test=test_area.copy()
df_test.loc[:,nameHot]=enc.transform(df_test[catt_col]).toarray()

for ar in train_area.area.unique():
    df333=df_test[df_test.area==ar].copy()
    test_pred_class=(models[ar].predict(df333[col_act_area+list(nameHot)])).astype(int)
    df333[targets]=test_pred_class
    allfortest.append(df333)
pdDF=pd.concat(allfortest)
df335=test_area.copy()
test_merge=pd.merge(df335,pdDF,how="left",on=["date","area"])
test_merge[['date', 'area'] +
                targets].to_csv(file_name+"/area.csv", index=False)
print("We are done forecasting Stations. a file area.csv is been saved in output folder.\n\n")
#clear memory 
del test_merge,df335,test_pred_class,pdDF,models,X,area_x_train,area_y_train,test_area,train_station_A,df7,index_rem
gc.collect()

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
############################        Global      ############################
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

#import
print("We will start importing train set and test set for Global")
train_station_g = pd.read_csv("data_to_use/remCharEWM4.csv", sep=",")
train_station_g['date'] = pd.to_datetime(train_station_g['date'])
train_station_g['Postcode'] = train_station_g['Postcode'].astype(str)
train_global = train_station_g.groupby('date').agg({'Available': 'sum',
                                                  'Charging': 'sum',
                                                  'Passive': 'sum',
                                                  'Other': 'sum',
                                                  'tod': 'max',
                                                  'dow': 'max',
                                                  'trend': 'max'}).reset_index()
#make cols
test_global= test_station.groupby('date').agg({
                                'tod': 'max',
                                'dow': 'max',
                                'trend': 'max',
                                'min': 'max',
                                'hour': 'max',
                                'quarter': 'max',
                                'dayofmonth': 'max',}).reset_index()
train_global=make_cols(train_global,False)
col_act_g=["hour","tod","min","dow","trend"]

x_train_g= train_global[col_act_g].copy()
y_train_g=train_global[targets].copy()
cat_features_index = [0,3]
########################################################
#model
print("Starting modelisation for Global... ")
clf_g = cb.CatBoostRegressor(eval_metric="MultiRMSE",loss_function="MultiRMSE",one_hot_max_size=200
                           ,depth= 3, iterations =1500 , learning_rate= .1,l2_leaf_reg= 1)
clf_g=(clf_g).fit(x_train_g,y_train_g, cat_features= cat_features_index,verbose=0)
print("Done modeling, now we are going to forecast.")
########################################################
#export
df3334=test_global.copy()
test_pred_class=np.floor(clf_g.predict(df3334[col_act_g])).astype(int)
df3334[targets]=test_pred_class
df3334[['date'] +
                  targets].to_csv(file_name+"/global.csv", index=False)
print("We are done forecasting Stations. a file global.csv is been saved in output folder.\n\n")
#clear memory 
del test_global,y_train_g, x_train_g,df3334,test_pred_class,test_station,train_global
gc.collect()

########################################################
#---------------------------------------------------------------------------------------
########################################################

print("Zipping",file_name, "folder...")
import shutil
shutil.make_archive(file_name,
                    'zip', file_name)
print(f"Done, your file {file_name}.zip is ready.\n")

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

