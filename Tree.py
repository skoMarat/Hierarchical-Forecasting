from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
from numpy.linalg import inv
import os
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime,timedelta
from scipy.interpolate import splrep, splev
import re
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.covariance import MinCovDet
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from pmdarima.arima import auto_arima
from pandas import to_datetime
from prophet import Prophet
from dateutil.relativedelta import relativedelta
from leaf import *
from utils import *
from forecast_prophet import *
from forecast_ucm import *
from forecast_sarimax import *
import copy
from collections import OrderedDict
from itertools import chain
from datetime import datetime
import pickle
import gc


class Tree:
    def __init__(self, dfData:pd.DataFrame , sType: str , iLeaf=None):
        """ 
        A tree object is a collection of leaf objects. 
        OnlyKPI and date is required for
        creation of whole tree according to listL4, listL3, listL2 specified below
        The order at which these lists are written is important. Stacking of reconciliation matrices is precise and is implied from dLeafs
                              
        Forecasts will be stored in matrix , shape of which is determined by the number of leafs in a given level.
        Inputs:   
                                 
        Outputs:
           mS: summation matrix, see getMatrixS() for more detail
           mP: projection matrix, see getMatrixP() for more detail
           mW: weight matrix, see getMatrixW() for more detail 
        Methods for Outputs:
           There are several methods that allows user to quickly and efficiently conduct diagnostics of the algorithm results:
             
        """       
        
        self.sType     = sType
        self.iLeaf     = iLeaf  #None if spatial m if temporal , then Tree is a leaf of a spatial Tree
        self.dfData    = dfData
        
                                             
        self.mS , self.levels , self.dLevels , self.list_of_leafs , self.date_time_index , self.sFreqData = self.get_mS()

        self.mY = self.get_mY()
       
        self.dForecasters = None # dictionary of forecast instances
        
        self.mP        = None
        self.mW        = None
        self.mYhat     = None
        self.mYhatIS   = None
        self.mYtilde   = None 
        self.mYtildeIS = None
        self.mRes      = None    # matrix that stores in sample base forecast errors.
        self.mResRec   = None  # matrix that stores in sample reconciled forecast errors
        
    def get_mY(self):
        """Puts data into a mY according to hierarchical nature
        for temporal hierarchy, subsets data so that all data values add up
        
        """
        mS=self.mS

        if self.sType=='spatial':
            df_data=self.dfData
            #create tree data matrix mY
            mY=np.zeros( (len(self.list_of_leafs), len(self.date_time_index)))
            for i,leaf_creds in enumerate(self.list_of_leafs):
                mY[i]=subset_data(df_data, self.levels,leaf_creds).values                                                                   
        elif self.sType=='temporal':
            if self.levels[-1]=='D': # works only for W and M data currently #TODO
                start_index = self.dfData[self.dfData.index.weekday == 0].index[0]
                end_index = self.dfData[self.dfData.index.weekday==6].index[-1]
            elif self.levels[-1]=='M':
                start_index = self.dfData[self.dfData.index.month == 1].index[0]
                end_index = self.dfData[self.dfData.index.month==12].index[-1]

            df_data=self.dfData[start_index:end_index]
            
            n=int(df_data.shape[0]/mS.shape[1])
            m=mS.shape[1]
            mYbottom=df_data.values.reshape((n,m)).T
            mY=mS@mYbottom
            
            self.dfData=df_data
            self.date_time_index=self.dfData.index
        return mY 
    
    def get_mS(self):
        if self.sType=='spatial':
            df_data = self.dfData
            #TODO change below to accomodate levels and prices
            levels = df_data.columns[pd.to_datetime(df_data.columns, errors='coerce').isna()] 
            date_time_index = pd.to_datetime(df_data.drop(columns=levels).columns)  
            sFreqData=date_time_index.inferred_freq                
            #create a hierarchy list
            df = df_data[levels]
            list_of_leafs = df.values.tolist()  
            
            for level in levels[::-1]:
                df[level]=None
                df=df.drop_duplicates()
                list_of_leafs.extend(df.values.tolist())  

            list_of_leafs=sorted(list_of_leafs, key=sort_key)
            
            dLevels={}
            for i,l in enumerate(levels):
                dLevels[levels[-i-1]]=len([sublist for sublist in list_of_leafs if sublist.count(None) == i]) #70
            dLevels['total']=1
            dLevels = OrderedDict(reversed(list(dLevels.items())))

            mS=np.ones((1,dLevels[levels[-1]])) # start with 1 row at the top of matrix s 
            #that is always a vector of ones of size equal to # of bottom level series
            for i,_ in enumerate(levels.to_list()):
                groupByColumns=levels.to_list()[:i+1]
                vBtmLevelSeries=df_data.groupby(groupByColumns).count().iloc[:,0].values
                mS=np.vstack([mS,create_cascading_matrix(vBtmLevelSeries)]) 
    
        elif self.sType=='temporal':
            #data will be a series not dataframe            
            levels=['A', 'SA', 'Q', 'M', 'W', 'D', 'H', 'T']
            dLevels={'A': 1, 'SA': 2 ,'Q': 2*2, 'M': 2*2*3, 'W': 1 , 'D':7 , 'H': 7*24 , 'T': 7*24*60}
            
            sFreqData=self.dfData.index.inferred_freq
            end_index = levels.index(sFreqData) #the bottom frequency, the freq of data
            start_index = 4 if end_index>3 else 0      
            
            levels=levels[start_index:end_index + 1] 
            dLevels={key: dLevels[key] for key in levels if key in dLevels.copy()}
                       
            mS=np.ones(dLevels[sFreqData],dtype=int) #starts from top level
        
            for i,sFreq in enumerate(levels[1:]):
                vBtmLevelSeries=np.full(dLevels[sFreq],int(dLevels[sFreqData]/dLevels[sFreq]))
                mS_=create_cascading_matrix(vBtmLevelSeries)
                mS=np.vstack(( mS, mS_ ))
                
            list_of_leafs=[]
            for i,sFreq in enumerate(levels):
                list_of_leafs.append([str(levels[i])+"_"+str(u) for u in np.arange(1,dLevels[sFreq]+1)])
            list_of_leafs = list(chain.from_iterable(list_of_leafs))
            
            date_time_index=None  # will be passed in getmY      
        return mS , levels, dLevels, list_of_leafs , date_time_index , sFreqData
                
    def getMatrixW(self , sWeightType:str):
        """
        Purpose:
        create the weights matrix
        
        MINT_method means that mW is a proxy for Sigma^-{inv}
        
        Outputs:
        mW:            matrix of weights
        """

        mRes=self.mRes.copy()
        if self.sModel=='ucm' or self.sModel=='sarimax':
             mRes=mRes[:,7*10:]  #ucm is state space, initial errors are too wild.

        # mRes=mRes[:,~np.all(np.isnan(mRes), axis=0)]
        mW = np.eye(mRes.shape[0])
        # vNonNanRows = np.setdiff1d(np.arange(0,mRes.shape[0]),  np.unique(np.argwhere(np.isnan(mRes))[:,0]))
        # mRes = mRes[vNonNanRows,:]
        n=mRes.shape[0]
        m=mRes.shape[1]
        # mSigma = (mRes@mRes.T)/m
                
        mRes_centered = mRes - np.mean(mRes, axis=1).reshape(n,1)
        mSigma = (mRes_centered@mRes_centered.T)/(m-1)
        
        if sWeightType == 'ols':  
            mW = np.eye(n) 
        elif sWeightType == 'wls':
            vW=np.loadtxt("c:\\Users\\31683\\Desktop\\data\\M5\\weights.txt")
            vW=vW[:self.mY.shape[0]]
            vW=vW/vW[0]
            mW=np.diag(vW) 
        elif sWeightType== 'wls_svar': #variance scaling of temporal
            vW=np.empty(0)
            for l,level in enumerate(self.levels):
                fVar=split_matrix(mRes_centered,self.dLevels)[l].var()
                vW=np.concatenate((vW,np.full(self.dLevels[level],fVar)))        
            mW=np.diag(vW)
        # elif sWeightType== 'wls_acov': #auto covariance scaling
        #     for l,level in enumerate(self.levels):
        #         vACOV_level=acovf(split_matrix(mRes_centered,self.dLevels)[l].flatten('F'))[1:]
                
                
        #         if l==0:
        #             mW=diag_mat([mSigma_level[0]])
        #         else:
        #             mW=diag_mat([mW,mSigma_level])   
        # elif sWeightType== 'wls_acov_nystrup': #auto covariance scaling
        #     for l,level in enumerate(self.levels):
        #         mRes_centered_level=split_matrix(mRes_centered,self.dLevels)[l]
        #         mSigma_level=(mRes_centered_level@mRes_centered_level.T)/(m-1)
                
        #         if l==0:
        #             mW=diag_mat([mSigma_level])
        #         else:
        #             mW=diag_mat([mW,mSigma_level])     
        elif sWeightType == 'wls_struc':  #structural scaling
            fVar=split_matrix(mRes_centered,self.dLevels)[-1].var()
            mW=fVar*np.diag(np.sum(self.mS, axis=1))  
        elif sWeightType == 'mint_diag':
            mW=np.diag(np.diag(mSigma))
            mW=np.linalg.inv(mW)  
        elif sWeightType == 'wls_var': #variance scaling of spatial 
            mW=np.diag(1/np.diag(mSigma))    
        elif sWeightType == 'mint_sample':  # full
            mW = np.linalg.inv(mSigma)   
        elif sWeightType == 'mint_shrink':
            mWF = mSigma.copy()
            mWD = np.diag(np.diag(mWF)) # all non-diagonal entries of mWF set to 0        

            sum_var_emp_corr = np.float64(0.0)
            sum_sq_emp_corr = np.float64(0.0)
            factor_shrinkage = np.float64(1 / (m * (m - 1)))

            for i in range(n):
                # Mean of the standardized residuals
                X_i =  mRes_centered[i]
                Xs_i = X_i / (np.std(mRes[i]) )
                Xs_i_mean = np.mean(Xs_i)
                for j in range(i + 1):
                    X_j = mRes_centered[j]
                    # Off-diagonal sums
                    if i != j:
                        Xs_j = X_j / (np.std(mRes[j]))
                        Xs_j_mean = np.mean(Xs_j)
                        # Sum off-diagonal variance of empirical correlation
                        w = (Xs_i - Xs_i_mean) * (Xs_j - Xs_j_mean)
                        w_mean = np.mean(w)
                        sum_var_emp_corr += np.sum(np.square(w - w_mean))
                        # Sum squared empirical correlation
                        sum_sq_emp_corr += w_mean**2

            # Calculate shrinkage intensity 
            dLambda = max(min((factor_shrinkage * sum_var_emp_corr) / (sum_sq_emp_corr ), 1.0), 0.0)
            mW = dLambda * mWD + (1-dLambda) * mWF
            mW = np.linalg.inv(mW) 
        
        self.mW=mW
    
    def getMatrixP(self , sWeightType: str , h=None):  
        """
        Purpose:
            return projection matrix P for a given mS and mW, as per Wickramasuriya et al (2018) equation (9)
            aka matrix G in some litterature such as as Hyndman et al. (2019).
            If sMeth is bottom up then mW is not used.  
        
        Return value:
            self.mP     projection matrix
        """
        
        mS=self.mS
        mW=self.mW
        
        if sWeightType == 'bottom_up':
            n=mS.shape[1]
            m=mS.shape[0]
            m0=np.full((n,m-n),0, dtype=int)
            mI=np.eye(n)
            mP=np.hstack((m0,mI))
            self.mP=mP
        elif 'top_down' in sWeightType:
            n=mS.shape[1]
            m=mS.shape[0]
            m0=np.full((n,m-1),0, dtype=int)
            if self.sType=='spatial':
                iB= len([sublist for sublist in self.list_of_leafs if sublist.count(None) == 0])# integer length of bottom level
            else:
                iB=7 #TODO
            if sWeightType=='top_down_hp': #historical proportions
                vP = np.mean((self.mY[-iB:]/self.mY[0,:]),axis=1)
                vP=vP.reshape((iB,1))
                mP=np.hstack((vP,m0))
                self.mP=mP  
            elif sWeightType=='top_down_ph': #proportions of the historical averages
                vP = np.mean(self.mY[-iB:] , axis=1)/np.mean(self.mY[0,:],axis=0)
                vP=vP.reshape((iB,1))
                mP=np.hstack((vP,m0))
                self.mP=mP  
            elif sWeightType=='top_down_ar':
                mP= self.mY[-70:]/self.mY[0,:]
                mP_diff=np.diff(mP,axis=1)
                num_rows = mP_diff.shape[0]
                mP_diff_forecast = np.zeros((num_rows, 8))
                mP_diff_forecast[:,0]=mP[:,-1]
                for i in range(0, num_rows):
                    model = AutoReg(mP_diff[i], lags=7).fit()
                    forecast = model.predict(start=len(mP_diff[i]), end=len(mP_diff[i]) + 6)
                    # Opslaan in mP_forecast
                    mP_diff_forecast[i, 1:] = forecast
                mP_forecast=mP_diff_forecast.cumsum(axis=1)[:,1:] 
                mP=np.hstack((mP_forecast[:,h].reshape((iB,1)),m0))
        else:
             mP = (np.linalg.inv(mS.T @ (mW @ mS)) @ (mS.T @ (mW)))
             self.mP = mP    
    
    # def tune_temporal_prophet(self, sTransform:str , iSize:int, iInitial:int, iPeriod:int, iHorizon:int, sMetric='rmse',
    #                           mX=None,dfHolidays=None, dfChangepoints=None ):
    #     """
    #     Tunes temporal and spatial series
    #     """
    #     for i,_ in enumerate(self.list_of_leafs):
    #         df_data = pd.DataFrame(data=self.mY[i] , index=self.date_time_index , columns=['y']) 
            
    #         tree_temporal=Tree(dfData=df_data, sType='temporal', iLeaf=i)
    #         tree_temporal.tune_prophet(sTransform=sTransform, iSize=iSize,
    #                                    iInitial=iInitial,
    #                                    iPeriod=iPeriod,
    #                                    iHorizon=iHorizon, sMetric=sMetric,mX=mX,
    #                                    dfHolidays=dfHolidays , dfChangepoints=dfChangepoints)
    
    def forecast_temporal_prophet(self, iOoS:int , sTransform:str , mX=None, dfHolidays=None, dfChangepoints=None ):
        """
        Forecast temporal prophet 
        iOoS : number of oos forecasts to generate for the spatial tree , number of iOoS at bottom level of temporal tree
        sTransfrom: a transform to apply to mY
        mX : matrix of exogenous variable
        dfHolidays:  dataframe of holidays to be passed to prophet
        dfChangepoint: dataframe of changepoints to be passed to prophet
        
        """   
       
        n=self.mY.shape[0]
        m=iOoS
        self.mYhat=np.zeros((n,m))
        self.dForecasters={}
    
        for i,_ in enumerate(self.list_of_leafs):
            df_data = pd.DataFrame(data=self.mY[i] , index=self.date_time_index , columns=['y']) 
            
            tree_temporal=Tree(dfData=df_data, sType='temporal' , iLeaf=i) 
            
            ############
            # l_methods_temporal=["bottom_up" , "top_down_ph" ,"top_down_hp","wls_svar" ,"wls_acov", "ols",
            #                "wls_struct" , "wls_hvar" , "mint_sample", "mint_shrink" , "mint_diag" ] 
            # ddOutputs=tree_temporal.cross_validation(sTransform=sTransform, dfHolidays=dfHolidays,
            #                                iInitial=int(tree_temporal.mY.shape[1]*0.9),
            #                                iPeriod=time_converter(iOoS, from_unit='D' , to_unit='W')*4,
            #                                iHorizon=time_converter(iOoS, from_unit='D' , to_unit='W'),
            #                                lMethods=l_methods_temporal, sForecastMethod='prophet'
            #                                )
            # df_temporal_cv_results=getCVResults(h=time_converter(iOoS, from_unit='D' , to_unit='W'), 
            #              iOoS=time_converter(iOoS, from_unit='D' , to_unit='W'),
            #              ddOutputs=ddOutputs, metric='RMSE', 
            #              slices=[7,1],  #TODO
            #              rolling=True, iters=None)
            # sSelectedWeightType = df_temporal_cv_results.loc['D'].idxmax()[0] #[0] compared to base
            # print(sSelectedWeightType)

            ##########################
            # ddISmatrices={}
            # l_methods_temporal=["bottom_up" , "top_down_ph" ,"top_down_hp","wls_svar" ,"wls_acov", "ols",
            #                "wls_struct" , "wls_hvar" , "mint_sample", "mint_shrink" , "mint_diag" ] 
          
            # tree_temporal.forecast_prophet(iOoS=7,sTransform=sTransform, mX=None, dfHolidays=dfHolidays)
            # for method in l_methods_temporal:
            #     tree_temporal.reconcile(method)
            #     ddISmatrices[method]={}
            #     ddISmatrices[method]['mYtilde']=tree_temporal.mYtildeIS
            #     ddISmatrices[method]['mYtrue']=tree_temporal.mY
            #     ddISmatrices[method]['mYhat']=tree_temporal.mYhat
            
            # df_is_results=getCVResults(h=time_converter(iOoS, from_unit='D' , to_unit='W'), 
            #              iOoS=time_converter(iOoS, from_unit='D' , to_unit='W'),
            #              ddOutputs=ddISmatrices, metric='MSE', 
            #              slices=[7,1],  #TODO
            #              rolling=True, iters=None)
            # sSelectedWeightType=df_is_results.loc['D'].idxmax()[0] #[0] compared to base
            # print(sSelectedWeightType)

            ################################
            
            if i==0:
                self.mYhatIS = np.zeros((self.mY.shape[0],tree_temporal.dfData.shape[0]))  #tree_temporal pottentially cuts data to fit the tree
                # self.mRes = np.zeros((self.mY.shape[0], self.mYhatIS.shape[1]))
            tree_temporal.forecast_prophet( iOoS=iOoS, sTransform=sTransform, dfHolidays=dfHolidays)
            
            sSelectedWeightType='bottom_up'
            if i >=40:
                sSelectedWeightType='top_down_ph'
            # sSelectedWeightType='top_down_ph'
            tree_temporal.reconcile(sSelectedWeightType)
            self.dForecasters[i]=tree_temporal.dForecasters
            self.mYhat[i] = split_matrix(tree_temporal.mYtilde,tree_temporal.dLevels)[-1].flatten(order='F')
            self.mYhatIS[i] = split_matrix(tree_temporal.mYtildeIS,tree_temporal.dLevels)[-1].flatten(order='F')
            # self.mYhatIS[i] = split_matrix(tree_temporal.mYhatIS,tree_temporal.dLevels)[-1].flatten(order='F')

                
        self.mRes=self.mYhatIS-self.mY[:,-self.mYhatIS.shape[1]:] #because some of mYhatIS might be missing
    
    # def tune_prophet(self, sTransform:str, iSize:int, iInitial:int, iPeriod:int, iHorizon:int, sMetric='rmse', 
    #                  mX =None , dfHolidays=None, dfChangepoints=None):       
    #     """
    #     Tunes prophet and saves parameters per leaf into ddParams
        
    #     """  
    #     if self.sType=='temporal':
    #         iIters=int((self.dfData.shape[0]-iInitial-iHorizon)/(iPeriod))+1
    #     elif self.sType=='spatial':
    #         iIters=int((self.mY.shape[1] - iInitial-iHorizon)/(iPeriod))+1
    #     print("Number of CV iterations used for tuning = " + str(iIters))
        
    #     try:
    #         with open("c:\\Users\\31683\\Desktop\\data\\M5\\ddParams_" + f"prophet_{sTransform}.pkl", 'rb') as file:
    #             dd_Params  = pickle.load(file) 
    #     except:
    #         dd_Params={}    
        
    #     if self.sType=='spatial':
    #         for iLeaf, _ in enumerate(self.list_of_leafs):
    #             if f"{iLeaf}_{self.sFreqData}" in dd_Params:
    #                 continue
    #             df_data = pd.DataFrame(data=self.mY[iLeaf] , index=self.date_time_index , columns=['y'])           
    #             pht = Forecast_Prophet(dfData=df_data, 
    #                                 dfX=mX[iLeaf] if mX is not None else None, 
    #                                 dfHolidays=dfHolidays,
    #                                 dfChangepoints=dfChangepoints)
    #             if sTransform is not None:
    #                 pht.transform(sTransform)    

    #             pht.tune(iSize=iSize,
    #                     iInitial=iInitial,
    #                     iPeriod=iPeriod,
    #                     iHorizon=iHorizon,
    #                     sMetric=sMetric, 
    #                     parallel='processes',
    #                     bPlot=False) 
                
    #             dd_Params[f"{iLeaf}_{self.sFreqData}"]=pht.dParams
    #     elif self.sType=='temporal':
    #         for i,sFreq in enumerate(self.levels):
    #             if f"{self.iLeaf}_{sFreq}" in dd_Params:
    #                 continue
    #             df_data=self.dfData.resample(sFreq).sum()
    #             df_data = pd.DataFrame(data=df_data.values , index=df_data.index , columns=['y'])
                    
    #             pht = Forecast_Prophet(dfData=df_data, dfX= mX[i] if mX is not None else None,  #TODO  for weekly data maybe other params?
    #                                 dfHolidays=dfHolidays, dfChangepoints=dfChangepoints)
                
    #             if sTransform is not None:
    #                 pht.transform(sTransform)
                
    #             pht.tune(iSize=iSize,
    #                      iInitial = time_converter(iInitial ,self.sFreqData, sFreq),
    #                      iPeriod  = time_converter(iPeriod  ,self.sFreqData, sFreq),
    #                      iHorizon = time_converter(iHorizon ,self.sFreqData, sFreq),
    #                      sMetric = sMetric, 
    #                      parallel = 'processes',
    #                      bPlot = False)  
    #             dd_Params[f"{self.iLeaf}_{sFreq}"]=pht.dParams  #X is a placeholder
    #     #save parameters
    #     file_path = "c:\\Users\\31683\\Desktop\\data\\M5\\ddParams_" + f"prophet_{sTransform}.pkl"
    #     with open(file_path, "wb") as myFile:
    #         pickle.dump(dd_Params, myFile)    
    
    # def forecast_prophet(self , iOoS:int, sTransform:str , mX=None, dfHolidays=None, dfChangepoints=None ): 
    #     """
    #     Performs the forecast algorithm at each leaf
    #     iOoS: of bottom level series

    #     """ 

    #     self.dForecasters={}
        
    #     #get parameters
    #     try:
    #         with open("c:\\Users\\31683\\Desktop\\data\\M5\\ddParams_" + f"prophet_{sTransform}.pkl", 'rb') as file:
    #             ddParams  = pickle.load(file) 
    #     except:
    #         ddParams={} 

    #     if self.sType=='temporal':           
    #         for i,sFreq in enumerate(self.levels): #start from lowest freq -> W
    #             df_data=self.dfData.resample(sFreq).sum()
    #             dfY = pd.DataFrame(data=df_data.values , index=df_data.index , columns=['y'])                   
    #             pht = Forecast_Prophet(dfData=dfY, dfX= mX[i] if mX is not None else None,
    #                                     dfHolidays=dfHolidays, dfChangepoints=dfChangepoints,
    #                                     dParams = ddParams[f"{self.iLeaf}_{sFreq}"] if bool(ddParams)!=False else None)  # if ddParams is populated then bool(ddParams) is True
                
    #             if sTransform is not None:
    #                 pht.transform(sTransform)
                    
    #             f=int(time_converter(iOoS, from_unit = self.levels[-1] , to_unit = sFreq))
    #             pht.forecast(iOoS=f)
                
    #             self.dForecasters[i]=pht  #save the forecaster into dict
                
    #             if sFreq==self.levels[0]: #if lowest freq -> W
    #                 self.mYhat = pht.vYhatOoS.reshape( 1, pht.vYhatOoS.shape[0])
    #                 self.mYhatIS = pht.vYhatIS.reshape( 1 , pht.vYhatIS.shape[0])
    #             else:
    #                 mYhat = pht.vYhatOoS.reshape( self.dLevels[sFreq],self.mYhat.shape[1] , order='F')
    #                 mYhatIS = pht.vYhatIS.reshape( self.dLevels[sFreq],self.mYhatIS.shape[1] , order='F')
                
    #                 self.mYhat=np.vstack((self.mYhat,mYhat)) 
    #                 self.mYhatIS = np.vstack((self.mYhatIS,mYhatIS))
                    
    #     elif self.sType=='spatial':
    #         self.mYhatIS = np.zeros((self.mY.shape[0], self.mY.shape[1]))
    #         self.mRes = np.zeros((self.mY.shape[0], self.mY.shape[1]))
                        
    #         n=self.mY.shape[0]
    #         m=iOoS
    #         self.mYhat=np.zeros((n,m))
            
    #         for i in range(self.mY.shape[0]): # start from leaf 0
    #             dfY = pd.DataFrame(data=self.mY[i] , index=self.date_time_index , columns=['y'])   
                            
    #             pht = Forecast_Prophet(dfData=dfY, dfX= mX[i] if mX is not None else None,
    #                                     dfHolidays=dfHolidays,dfChangepoints=dfChangepoints,
    #                                     dParams = ddParams[f"{i}_{self.sFreqData}"] if bool(ddParams)!=False else None)
                
    #             if sTransform is not None:
    #                 pht.transform(sTransform)
                
    #             pht.forecast(iOoS=iOoS)
    #             self.dForecasters[i]=pht
    #             self.mYhat[i] = pht.vYhatOoS
    #             self.mYhatIS[i] = pht.vYhatIS                                       
                        
    #     self.mRes=self.mYhatIS-self.mY    
     
    def forecast(self, iOoS:int , sModel, dfHolidays:pd.DataFrame , dfSNAP:pd.DataFrame, dfPrice:pd.DataFrame , sTempRecMethod:str ): 
        """
        dfSNAP : univariate time series of SNAP
        sModel: 'ucm', "sarimax'
        
        """            
        self.sModel=sModel
        self.dForecasters={}

        if self.sType=='temporal':      
            for i,sFreq in enumerate(self.levels): #start from lowest freq -> W               

                # dfY=self.dfData.resample(sFreq).sum(min_count=1)
                #preserve nans at the end (oOS)
                dfY=self.dfData.resample(sFreq).apply(lambda x: x.sum() if not x.isna().all() else np.nan)
                
                
                dfP=dfPrice.resample(sFreq).mean()
                dfS=dfSNAP.resample(sFreq).sum()
                dfH=dfHolidays.resample(sFreq).sum()
                df=pd.concat([dfY,dfP,dfH,dfS],axis=1)
                
                if sModel=='ucm':
                    model = Forecast_UCM(dfData=df)
                    model.transform('log')
                    f=int(time_converter(iOoS, from_unit = self.levels[-1] , to_unit = sFreq))
                    model.fit()
                    model.forecast(iOoS=f)
                    model.transform('log')
                elif sModel=='sarimax':
                    model = Forecast_SARIMAX(dfData=df)
                    f=int(time_converter(iOoS, from_unit = self.levels[-1] , to_unit = sFreq))
                    model.fit()
                    model.forecast(iOoS=f)
                     

                self.dForecasters[i]=model
                
                if sFreq==self.levels[0]: #if lowest freq -> W
                    self.mYhat = model.vYhatOoS.reshape( 1, model.vYhatOoS.shape[0])
                    self.mYhatIS = model.vYhatIS.reshape( 1 , model.vYhatIS.shape[0])
                else:
                    mYhat = model.vYhatOoS.reshape( self.dLevels[sFreq],self.mYhat.shape[1] , order='F')
                    mYhatIS = model.vYhatIS.reshape( self.dLevels[sFreq],self.mYhatIS.shape[1] , order='F')
                
                    self.mYhat=np.vstack((self.mYhat,mYhat)) 
                    self.mYhatIS = np.vstack((self.mYhatIS,mYhatIS))
                del model
                gc.collect()
                    
        elif self.sType=='spatial':
            self.mYhatIS = np.zeros((self.mY.shape[0], self.mY.shape[1]))
            self.mRes = np.zeros((self.mY.shape[0], self.mY.shape[1]))
                        
            n=self.mY.shape[0]
            m=iOoS
            self.mYhat=np.zeros((n,m))
            
            tree_price=Tree(dfData = dfPrice , sType='spatial')
            tree_snap=Tree(dfData = dfSNAP , sType= 'spatial') 
            
            
            for i in range(self.mY.shape[0]): # start from leaf 0
                df=pd.DataFrame(
                    data={'price': tree_price.mY[i, :], 'snap': tree_snap.mY[i, :]},
                    index=tree_price.date_time_index[:] )
                df.snap=(df.snap!=0).astype(int)
                
                dfHolidays.loc[(dfHolidays.date.dt.day != 25) & (dfHolidays.date.dt.month != 12), "holidays"] = 1
                dfHolidays.loc[(dfHolidays.date.dt.day == 25) & (dfHolidays.date.dt.month == 12), "christmas"] = 1
                dfHolidays.set_index('date')
                dfHolidays[['holidays','christmas']]=dfHolidays[['holidays','christmas']].fillna(0).astype(int)
                
                df=pd.merge(df,dfHolidays, left_index=True,  right_on='date', how='left' )
                df=df.fillna(0)
                df[['holidays','christmas']]=df[['holidays','christmas']].astype(int)

                dfY=pd.DataFrame(data=self.mY[i,:], index=self.date_time_index[:] , columns=['y'])
                df.set_index('date',inplace=True)
                df=pd.merge(df,dfY, left_index=True,  right_index=True, how='left' )      
                              
                if sTempRecMethod is not None:   # it is temporal UCM
                    tree_temporal=Tree(dfData=df[['y']].dropna(), sType='temporal' , iLeaf=i)
                    
                    if sModel=='ucm':
                        tree_temporal.forecast(iOoS=iOoS ,sModel=sModel, 
                                               dfHolidays=df[['holidays','christmas']],
                                               dfSNAP=df[['snap']] , dfPrice=df[['price']],
                                               sTempRecMethod=sTempRecMethod)
                    elif sModel=='sarimax':
                        tree_temporal.forecast(iOoS=iOoS ,sModel=sModel, 
                                               dfHolidays=df[['holidays','christmas']],
                                               dfSNAP=df[['snap']] , dfPrice=df[['price']],
                                               sTempRecMethod=sTempRecMethod)
                    
                    tree_temporal.reconcile(sTempRecMethod)
                    
                    self.dForecasters[i]=tree_temporal.dForecasters
                    self.mYhat[i] = split_matrix(tree_temporal.mYtilde,tree_temporal.dLevels)[-1].flatten(order='F')
                    # self.mYhatIS[i] = split_matrix(tree_temporal.mYtildeIS,tree_temporal.dLevels)[-1].flatten(order='F')
                    self.mYhatIS[i] = split_matrix(tree_temporal.mYhatIS,tree_temporal.dLevels)[-1].flatten(order='F')

                else:
                    if sModel=='ucm':
                        model = Forecast_UCM(dfData=df)
                        model.transform('log')
                        model.fit()
                        model.forecast(iOoS=iOoS)
                        model.retransform()
                    
                    elif sModel=='sarimax':
                        model = Forecast_SARIMAX(dfData=df)
                        model.fit()
                        model.forecast(iOoS=iOoS)
                    
                    self.dForecasters[i]=model
                    self.mYhat[i] = model.vYhatOoS
                    self.mYhatIS[i] = model.vYhatIS 
                    del model
                    gc.collect()
                          
        self.mRes=self.mYhatIS-self.mY[:,-self.mYhatIS.shape[1]:] #because some of mYhatIS might be missing                     
    
    def reconcile(self , sWeightType: str):
        """
        Performs whole reconciliation algorithm 
        """                                     
        if sWeightType=='top_down_ar' :
           self.mYtilde=np.zeros((self.mYhat.shape[0],self.mYhat.shape[1])) 
           for h in range(0, self.mYhat.shape[1]):
              self.getMatrixP(sWeightType,h=h)
              self.mYtilde[:,h]=np.dot(np.dot(self.mS,self.mP),self.mYhat[:,h])
        else:       
            self.getMatrixW(sWeightType)      
            self.getMatrixP(sWeightType) 
            self.mYtilde=np.dot(np.dot(self.mS,self.mP),self.mYhat)
            self.mYtildeIS=np.dot(np.dot(self.mS,self.mP),self.mYhatIS)
            # self.mResRec=self.mYtildeIS-self.mYhat
            
    
    def cross_validation(self ,  sTransform:str , dfHolidays:pd.DataFrame,  dfSNAP:pd.DataFrame , dfPrice: pd.DataFrame,
                         iInitial:int, iPeriod:int, iHorizon:int, lMethods:list,
                         sForecastMethod:str , sTempRecMethod:str):
        """Performs cross_validation and returns matrices required for assesment

        Args:
            initial (_type_):  if spatial, iOoS required, if spatial iOoS of lowest freq
            period (_type_): _description_
            horizon (_type_): _description_
            sWeightType (_type_): _description_

        Returns:
            mYhat (np.darray) :      base forecasts matrix of size nx(m*h) such that n is the number of leafs, m is the number of iterations of CV and h is the horizon
                                     each iteration is horizontally stacked such that first h is the first iteration base forecasts
            mYtilde (np.darray) :    similar to mYhat but with reconciled forecasts
        """
        # X_path=os.getcwd()+f"\\data\\M5\\prices_train_val_eval.csv"  # to data file
        # mX  = get_mX(X_path)
        if self.mYhat is not None:
            print("cross_validation can only be performed on initiated Tree object")
            return
        
        iIters=int((self.mY.shape[1] - iInitial-iHorizon)/(iPeriod))+1
        print("Number of CV folds = " + str(iIters))
 
        dOutputs={}
        
        for method in lMethods:
            dOutputs[method]={}  #TODO can be moved inside of \\for sWeightType in lMethods:\\  loop
            
        for iter in range(iIters):  
            if iter>50:
                continue
            tree_iter = copy.copy(self) 
            tree_iter.mY = self.mY[:, 0+iPeriod*iter : iInitial+iPeriod*iter ]
            tree_iter.date_time_index=self.date_time_index[ 0+iPeriod*iter : iInitial+iPeriod*iter ]

            if self.sType=='temporal':
                tree_iter.dfData=tree_iter.dfData.iloc[0+time_converter(iPeriod,'W','D')*iter : time_converter(iInitial , 'W','D')+time_converter(iPeriod,'W','D')*iter]
            tree_iter_eval = copy.copy(self)
            tree_iter_eval.mY = self.mY[:, iPeriod*iter+iInitial : iPeriod*iter+iInitial+iHorizon ]  
            
            #check if we are at the end of CV ( when there is not enough mY to fit iHorizon ammount of forecasts)
            if tree_iter_eval.mY.shape[1]!=iHorizon:
                break  
 
            # tree_iter.tune_temporal_prophet()
            if sForecastMethod=='prophet':
                iOoS=iHorizon if self.sType=='spatial' else time_converter(iHorizon, from_unit='W' , to_unit='D')
                tree_iter.forecast_prophet(iOoS=iOoS, 
                                           sTransform=sTransform,
                                           dfHolidays=dfHolidays)
            elif sForecastMethod=='temporal_prophet':
                tree_iter.forecast_temporal_prophet(iOoS=iHorizon ,
                                                    sTransform=sTransform , 
                                                    mX=None, 
                                                    dfHolidays=dfHolidays,
                                                    dfChangepoints=None )
            elif sForecastMethod=="ucm":
                iOoS=iHorizon if self.sType=='spatial' else time_converter(iHorizon, from_unit='W' , to_unit='D')
                tree_iter.forecast(iOoS=iOoS , sModel='ucm',
                      dfHolidays=dfHolidays,
                      dfSNAP=dfSNAP,
                      dfPrice=dfPrice, sTempRecMethod=None)
            elif sForecastMethod=="sarimax":
                iOoS=iHorizon if self.sType=='spatial' else time_converter(iHorizon, from_unit='W' , to_unit='D')
                tree_iter.forecast(iOoS=iOoS , sModel='sarimax',
                      dfHolidays=dfHolidays,
                      dfSNAP=dfSNAP,
                      dfPrice=dfPrice,sTempRecMethod=None)
            elif sForecastMethod=="temporal_ucm":
                iOoS=iHorizon if self.sType=='spatial' else time_converter(iHorizon, from_unit='W' , to_unit='D')
                tree_iter.forecast(iOoS=iOoS , sModel='ucm',
                      dfHolidays=dfHolidays,
                      dfSNAP=dfSNAP,
                      dfPrice=dfPrice, sTempRecMethod=sTempRecMethod)
            elif sForecastMethod=="temporal_sarimax":
                iOoS=iHorizon if self.sType=='spatial' else time_converter(iHorizon, from_unit='W' , to_unit='D')
                tree_iter.forecast(iOoS=iOoS , sModel='sarimax',
                      dfHolidays=dfHolidays,
                      dfSNAP=dfSNAP,
                      dfPrice=dfPrice,sTempRecMethod=sTempRecMethod)
            
            for sWeightType in lMethods:            
                tree_iter.reconcile(sWeightType)  
                
                if iter!=0:
                    dOutputs[sWeightType]['mYtrue'] = np.hstack([dOutputs[sWeightType]['mYtrue'] , tree_iter_eval.mY[:,-iHorizon:]])
                    dOutputs[sWeightType]['mYhat'] = np.hstack([dOutputs[sWeightType]['mYhat'] , tree_iter.mYhat[:,-iHorizon:]])
                    dOutputs[sWeightType]['mYtilde'] = np.hstack([dOutputs[sWeightType]['mYtilde'] , tree_iter.mYtilde[:,-iHorizon:]])
                else:
                    dOutputs[sWeightType]['mYtrue'] = tree_iter_eval.mY[:,-iHorizon:]
                    dOutputs[sWeightType]['mYhat'] = tree_iter.mYhat[:,-iHorizon:]  #TODO is there need for horizon here?
                    dOutputs[sWeightType]['mYtilde'] = tree_iter.mYtilde[:,-iHorizon:] #TODO is there need for horizon here?
                    dOutputs[sWeightType]['mW']=tree_iter.mW
                    
            if self.sType == 'spatial':
                print("CV iterations completed = " + str(iter+1) + " of " + str(iIters))
        
        return dOutputs 

        