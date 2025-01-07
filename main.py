import pandas as pd
from importlib import reload
import numpy as np
import os

# import simulation
# reload(simulation)
# from simulation import *

import leaf
reload(leaf)
from leaf import *

import forecast_arima
reload(forecast_arima)
from forecast_arima import *

import Tree
reload(Tree)
from Tree import *

import forecast_prophet
reload(forecast_prophet)
from forecast_prophet import *

import utils
reload(utils)
from utils import *

import logging
logging.getLogger('cmdstanpy').addHandler(logging.NullHandler())
        

def prepare_weights(path:str):
    """
    saves weights to data folder
    path (str) : location of data 
    
    """
    #reads data that was downloaded from Kaggle into desktop as a csv file
    dfValidation=pd.read_pkl(path+"\\sales_train_validation.pkl") 
    dfCalendar=pd.read_csv(path+"\\calendar.csv")
    dfPrices=pd.read_pkl(path+"\\sell_prices.pkl")
    
    df=pd.concat([dfValidation[['store_id','item_id']],dfValidation.iloc[:,-28:]], axis=1)
    dfDW=dfCalendar[dfCalendar.d.isin(df.columns[1:])][['d','wm_yr_wk']]
    rename_dict = dict(zip(dfDW['d'], dfDW['wm_yr_wk']))
    df.rename(columns=rename_dict, inplace=True)
    df=df.groupby(level=0,axis=1).sum()
    df=pd.melt(df, id_vars=['store_id','item_id'], var_name='wm_yr_wk', value_name='cum_sum_sales')
    dfPrices=dfPrices[dfPrices.wm_yr_wk.isin(dfDW['wm_yr_wk'])][['store_id','item_id','wm_yr_wk','sell_price']]
    dfWeights=df.merge(dfPrices, on=["store_id", "item_id","wm_yr_wk"])
    dfWeights['weights']=dfWeights.cum_sum_sales*dfWeights.sell_price
    dfWeights=dfWeights.groupby(['store_id','item_id'])['weights'].sum().reset_index()
    dfWeights[['state_id', 'store_id']] = dfWeights['store_id'].str.split('_', expand=True)
    dfWeights[['cat_id', 'dept_id','item_id']] = dfWeights['item_id'].str.split('_', expand=True)
    dfWeights=dfWeights[['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id','weights']]
    
    #goal: create vW of weights , vW of size 42,840  - if grouped, but we are not doing grouped now , so 30490 + 
    levels=dfWeights.columns.drop('weights')
    df=dfWeights[levels]
    list_of_leafs=df.values.tolist()

    for level in levels[::-1]:
        df[level]=None
        df=df.drop_duplicates()
        list_of_leafs.extend(df.values.tolist())

    def sort_key(item):
        none_count = item.count(None)
        return (-none_count, item)

    list_of_leafs=sorted(list_of_leafs, key=sort_key)
    vW=np.zeros(len(list_of_leafs))


    def subset_data(data, levels,l:list):
        """
        subsets data to include only data of a certain leaf
        leaf_list (list)  size n, [0] is the level 0 while [-1] is the lowest level
        
        returns: serried of aggregated values for a given leaf_list
        """
        column_mask=(data[levels]==l).any(axis=0)  
        row_mask=(data[levels]==l).loc[:,column_mask].all(axis=1)
        
        srY=data[row_mask].drop(columns=levels).sum(axis=0)
        return srY 


    for i,leaf_creds in enumerate(list_of_leafs):
        vW[i]=subset_data(dfWeights,levels,leaf_creds).values
    
    np.savetxt(path+"\\weights.txt", vW, fmt='%d', delimiter=' ')  

def prepare_prices(path):
    """
    Method unique to Wallmart data, disaggregates sell_prices.pkl into lowest levels and
    creates a time series dataframe similar (or same) to sales_train_validation.csv
    further to be fed into get_mY method
    
    saves to data
    """
    dfPrices=pd.read_pkl(path+"\\sell_prices.csv")
    dfCalendar=pd.read_csv(path+"\\calendar.csv")
    dfPrices=dfPrices.pivot(index=['store_id','item_id'],columns='wm_yr_wk', values='sell_price').reset_index()
    df=dfPrices.iloc[:,:2]
    for week_number in dfPrices.columns:
        if week_number in dfCalendar['wm_yr_wk'].values:
            dates=dfCalendar[dfCalendar['wm_yr_wk']==week_number]['date']
            for date in dates:
                df[date]=dfPrices[week_number].values
                
    df[['state_id', 'store_id']] = df['store_id'].str.split('_', expand=True)
    df[['cat_id', 'dept_id','item_id']] = df['item_id'].str.split('_', expand=True)
    hierarchy_cols= df[['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id']]
    data_cols = df.loc[:, ~df.columns.isin(['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id'])]
    df=pd.concat([hierarchy_cols, data_cols], axis=1)
    
    df.to_pickle(path+f"\\data\\M5\\prices_train_val_eval.pkl", index=False)

def get_mX(path):
    """Puts price data into a mX according to hierarchical structure, propogates up the structure using mean    
    """
    data=pd.read_pickle(path, compression='gzip') 
    #based on data, find all possible levels and datetime index
    #TODO change below to accomodate levels and prices
    levels=data.columns[pd.to_datetime(data.columns, errors='coerce').isna()] 
    date_time_index=pd.to_datetime(data.drop(columns=levels).columns)                  
    #create a hierarchy list
    df=data[levels]
    list_of_leafs=df.values.tolist()  
    
    for level in levels[::-1]:
        df[level]=None
        df=df.drop_duplicates()
        list_of_leafs.extend(df.values.tolist())
        
    def sort_key(item):
            none_count = item.count(None)
            return (-none_count, item)       


    list_of_leafs=sorted(list_of_leafs, key=sort_key)
    
    #create tree data matrix mY
    mX=np.zeros( (len(list_of_leafs), len(date_time_index)))
    
    def subset_data(data,levels,l:list):
        """
        subsets data to include only data of a certain leaf
        leaf_list (list)  size n, [0] is the level 0 while [-1] is the lowest level
        
        returns: serried of aggregated values for a given leaf_list
        """
        column_mask=(data[levels]==l).any(axis=0)  
        row_mask=(data[levels]==l).loc[:,column_mask].all(axis=1)
        
        srY=data[row_mask].drop(columns=levels).mean(axis=0)
            
        return srY
    
    for i,leaf_creds in enumerate(list_of_leafs):
        mX[i]=subset_data(data, levels,leaf_creds ).values  
    
    return mX

    
def main():
    """
    Requirements:
        Python 3.9.13 is a requirement for skicit-fda 
        Python 3.7 or higher is required for Prophet
    
    Code starts with importing data via import_data()
    
    Code initiliazes Tree object for a given KPI and date, where date is the 3 month ahead date
       Initialized tree, initializes 78 Leafs , for each hiearchy level
       Each leaf is populated with its own data and covid_data
       Each leaf is capable of forecasting itself.
            When forecasting a leaf, the data of the leaf is transformed into a function data object
            fda object stores the data, partially observed curve, principal component scores and vectors.
            fda object is capable of demeaned and remeaning itself. 
            fda object is capable of smoothing itself via smooth_data(n_basis:int) method
            fda object is capable of extracting PC scores and vectors via perform_FPCA() method
       Each leaf is equipped with methods that can plot the data and the forecasts
       It can plot forecasts of beta, forecasts of the curve, reconciled forecasts and plot the data of the leaf    
    A tree has a method to reconcile itself given a reconciliation method
    When reconciliation is started, a forecast command is passed to each tree
    After each tree is forecasted, Tree  matrix objects gets populated (mS, mW, mP, mYhat)
    Thereafter, reconciliation is performed. You can reconcile a tree only once. To try a different parameter for reconciliation, you need to reinitialize the tree
    After reconciliation, each tree gets populated with reconclied forecasts, completing the whole algorithm
    A tree is equipped with plot_errors() method that creates an interactive plot of percentage errors per leaf (reconciled forecasts and base forecasts) 
    """
    path='c:\\Users\\31683\\Desktop\\data\\M5'
    Y_path=path+f"\\sales_train_validation.pkl"  # to data file  
    prepare_prices(path)
    X_path=os.getcwd()+f"\\prices_train_val_eval.pkl"  # to data file
    prepare_weights(path)
    
    ####################################
    weight_type = "diag"
    weight_type = "mint_shrink" 
    weight_type = "full"
    weight_type = "bottom_up"
    weight_type = "top_down_hp"
    weight_type = "top_down_ph"
    weight_type = "ols"
    ####################################
    
    iOoS=28  # at the bottom forecast frequency if temporal
    
    
    #additional (optional) data for Prophet forecasting
    mX  = get_mX(X_path)
    holidays=pd.read_csv(path+f"\\holidays.csv")
    holidays=holidays.values.flatten()
    # changepoints=pd.read_csv(os.getcwd()+f"\\data\\M5\\changepoints.csv")
    # changepoints=changepoints.values.flatten()
    
    #weights for M5 competition testing
    vW=np.loadtxt(path+f"\\weights.txt")
    vW=vW[:114]
    
    #start of the algorithm for a given weight
    tree=Tree( data_directory=Y_path, type='spatial')
    tree.forecast_Prophet(iOoS=iOoS, mX=mX[:,:-iOoS], holidays=holidays  )
    tree.reconcile( sWeightType = weight_type)    
        
      
                
    
    
if __name__ == "__main__":
    main()