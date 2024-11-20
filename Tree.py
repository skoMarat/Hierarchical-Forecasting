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
import copy
from itertools import chain
from datetime import datetime
import pickle


class Tree:
    def __init__(self, data:pd.DataFrame , type: str ):
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
            (1) getLeaf() method allows user to specify strings for Level and get the leaf object. 
                          Thus no need to look up which leafs corresponds to which tree in dLeafs
            (2) plot_errors() methods plots interactive plotly graph of errors per leaf         
            (3) 
             
        """       
        
        #
        self.data=data
        self.type=type
                                             
        self.mS , self.levels , self.dLevels , self.list_of_leafs , self.date_time_index=self.get_mS()

        self.mY = self.get_mY()
        
        self.ddParams = None  #forecasting parameters dictionary of dictionaries   
        self.dForecasters = None # dictionary of forecast instances
        
        self.mP    = None
        self.mW    = None
        self.mYhat = None
        self.mYhatIS = None
        self.mYtilde = None 
        self.mRes = None    # matrix that stores in sample base forecast errors.
        
    def get_mY(self):
        """Puts data into a mY according to hierarchical nature
        for temporal hierarchy, subsets data so that all data values add up
        
        """
        mS=self.mS

        if self.type=='spatial':
            dfData=self.data
            #create tree data matrix mY
            mY=np.zeros( (len(self.list_of_leafs), len(self.date_time_index)))
            for i,leaf_creds in enumerate(self.list_of_leafs):
                mY[i]=subset_data(dfData, self.levels,leaf_creds).values                                                                   
        elif self.type=='temporal':
            if self.levels[-1]=='D': # works only for W and M data currently #TODO
                start_index = self.data[self.data.index.weekday == 0].index[0]
                end_index = self.data[self.data.index.weekday==6].index[-1]
            elif self.levels[-1]=='M':
                start_index = self.data[self.data.index.month == 1].index[0]
                end_index = self.data[self.data.index.month==12].index[-1]

            data=self.data[start_index:end_index]
            
            n=int(data.shape[0]/mS.shape[1])
            m=mS.shape[1]
            mYbottom=data.values.reshape((n,m)).T
            mY=mS@mYbottom
            
            self.data=data
        return mY 
    
    def get_mS(self):
        if self.type=='spatial':
            data=self.data
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

            list_of_leafs=sorted(list_of_leafs, key=sort_key)
            
            dLevels={}
            for i,l in enumerate(levels):
                dLevels[levels[-i-1]]=len([sublist for sublist in list_of_leafs if sublist.count(None) == i]) #70
            dLevels['total']=1
            
            mS=np.ones((1,next(iter(dLevels.values())))) # start with 1 row at the top of matrix s 
            #that is always a vector of ones of size equal to # of bottom level series
            for i,_ in enumerate(levels.to_list()):
                groupByColumns=levels.to_list()[:i+1]
                vBtmLevelSeries=data.groupby(groupByColumns).count().iloc[:,0].values
                mS=np.vstack([mS,create_cascading_matrix(vBtmLevelSeries)])  
        else:
            #data will be a series not dataframe            
            levels=['A', 'SA', 'Q', 'M', 'W', 'D', 'H', 'T']
            dLevels={'A': 1, 'SA': 2 ,'Q': 2*2, 'M': 2*2*3, 'W': 1 , 'D':7 , 'H': 7*24 , 'T': 7*24*60}
            
            sFreqData=self.data.index.inferred_freq
            end_index = levels.index(sFreqData) #the bottom frequency, the freq of data
            start_index = 4 if end_index>3 else 0      
            
            levels=levels[start_index:end_index + 1] 
            dLevels={key: dLevels[key] for key in levels if key in dLevels.copy()}
                       
            mS=np.ones(dLevels[sFreqData],dtype=int) #starts from top level
            print(levels)
        
            for i,sFreq in enumerate(levels[1:]):
                vBtmLevelSeries=np.full(dLevels[sFreq],int(dLevels[sFreqData]/dLevels[sFreq]))
                mS_=create_cascading_matrix(vBtmLevelSeries)
                mS=np.vstack(( mS, mS_ ))
                
            list_of_leafs=[]
            for i,sFreq in enumerate(levels):
                list_of_leafs.append([str(levels[i])+"_"+str(u) for u in np.arange(1,dLevels[sFreq]+1)])
            list_of_leafs = list(chain.from_iterable(list_of_leafs))
            
            date_time_index=None  # will be passed in getmY      
        return mS , levels, dLevels, list_of_leafs , date_time_index
                
    def getMatrixW(self , sWeightType:str):
        """
        Purpose:
        create the weights matrix
        
        MINT_method means that mW is a proxy for Sigma^-{inv}
        
        Outputs:
        mW:            matrix of weights
        """

        mRes=self.mRes.copy()
        # mW = np.eye(mRes.shape[0])
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
            mW=np.diag(np.hstack((np.ones(n-70),np.zeros(70)))) 
        elif sWeightType== 'mint_svar': #variance scaling
            vW=np.empty(0)
            for l,level in enumerate(self.levels):
                fVar=split_matrix(mRes_centered,self.dLevels)[l].var()
                vW=np.vstack((vW,np.full(self.dLevels[level],fVar)))        
            mW=np.linalg.inv(np.diag(vW))       
        elif sWeightType== 'mint_acov': #auto covariance scaling
            for l,level in enumerate(self.levels):
                mRes_centered_level=split_matrix(mRes_centered,self.dLevels)[l]
                mSigma_level=(mRes_centered_level@mRes_centered_level.T)/(m-1)
                
                if l==0:
                    mW=diag_mat([mSigma_level])
                else:
                    mW=diag_mat([mW,mSigma_level])
            mW=np.linalg.inv(mW)       
        elif sWeightType == 'mint_struc':  #structural scaling
            mW=np.diag(np.sum(self.mS, axis=1))
            mW=np.linalg.inv(mW)   
        elif sWeightType == 'mint_diag' or sWeightType == 'mint_hvar': #variance scaling  or wls
            mW=np.diag(np.diag(mSigma))
            mW=np.linalg.inv(mW)      
        elif sWeightType == 'mint_sample':  # full
            mW = np.linalg.inv(mSigma)   
        elif sWeightType == 'mint_shrink':
            mWF = mSigma.copy()
            mWD = np.diag(np.diag(mWF)) # all non-diagonal entries of mWF set to 0
            
            # #calculate numerator
            # dBottom = 0 # lower side in the expression for tuning parameter lambda
            # for i in range(n):
            #     for j in range(n):
            #         if i>j:
            #             dBottom = dBottom + 2*( mWF[i,j] / np.sqrt(mWF[i,i]*mWF[j,j]) )            
            # #Calculate denominator            
            # mResScaled = mRes_centered.T / np.sqrt(np.diag(mWF)) # elementwise division, standardize residuals
            # mResScaledSq = mResScaled**2  
            # mUp = (1/(m*(m-1))) * ( (mResScaledSq @ mResScaledSq.T)- (1/m)*((mResScaled @ mResScaled.T)**2) )  
            
            # # mResScaledSq = mResScaled**2  #w_ii 
            # # mUp = (1/(m*(m-1))) * ( (mResScaledSq.T @ mResScaledSq)- (1/m)*((mResScaled.T @ mResScaled)))**2   #w_ii-w_bar 
          
        
            # dUp = 0 # lower side in the expression for tuning parameter lambda
            # for i in range(n):
            #     for j in range(n):
            #         if i>j:
            #             dUp = dUp + 2*mUp[i,j]
            
            # dLambda = np.max((np.min((dUp/dBottom, 1)), 0))           

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
    
    def getMatrixP(self , sWeigthType: str):  
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
        
        if sWeigthType == 'bottom_up':
            n=mS.shape[1]
            m=mS.shape[0]
            m0=np.full((n,m-n),0, dtype=int)
            mI=np.eye(n)
            mP=np.hstack((m0,mI))
            self.mP=mP
        elif 'top_down' in sWeigthType:
            n=mS.shape[1]
            m=mS.shape[0]
            m0=np.full((n,m-1),0, dtype=int)
            iB= len([sublist for sublist in self.list_of_leafs if sublist.count(None) == 0])# integer length of bottom level
            if sWeigthType=='top_down_hp': #historical proportions
                #TODO 70 dynamic
                vP = np.mean((self.mY[-iB:]/self.mY[0,:]),axis=1)
            elif sWeigthType=='top_down_ph': #proportions of the historical averages
                vP = np.mean(self.mY[-iB:],axis=1)/np.mean(self.mY[0,:],axis=0)
            # elif sWeigthType=='top_down_fp': #forecast proportions
            #     vP = 
            vP=vP.reshape((iB,1))
            mP=np.hstack((vP,m0))
            self.mP=mP    
        # else:
        #     mWinv = np.linalg.inv(mW)
        #     mP= (np.linalg.inv(mS.T @ (mWinv @ mS)) @ (mS.T @ (mWinv)))
        #     self.mP=mP
        else:
             mP = (np.linalg.inv(mS.T @ (mW @ mS)) @ (mS.T @ (mW)))
             self.mP=mP    
    
    def tune_Prophet(self, random_size=100,initial=1548,period=28,horizon=28,metric='rmse' , mX =None , 
                     dfHolidays=None, dfChangepoints=None):       
        """
        Tunes prophet and saves parameters per leaf into ddParams
        Initial = 1548 = (mY.shape[1]-365)*0.7
        
        """  
        iIters=int((self.mY.shape[1] - initial-horizon)/(period))+1
        print("Number of CV iterations used for tuning = " + str(iIters))
        
        self.ddParams={}
        for i, _ in enumerate(self.list_of_leafs):
            dfData = pd.DataFrame(data=self.mY[i] , index=self.date_time_index , columns=['y'])           
            pht = Forecast_Prophet(dfData=dfData, 
                                   dfX=mX[i] if mX is not None else None, 
                                   dfHolidays=dfHolidays,
                                   dfChangepoints=dfChangepoints)

            pht.tune(random_size=random_size,
                    initial=initial,
                    period=period,
                    horizon=horizon,
                    metric=metric, 
                    parallel='processes',
                    plot=False) 
            
            self.ddParams[i]=pht.dParams
        #save parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        with open(os.path.join(os.getcwd(), f"data\\M5\\ddParams_{timestamp}.pkl"), "wb") as myFile:
            pickle.dump(self.ddParams, myFile) 
                
    def forecast_Prophet(self , iOoS:int, mX=None, dfHolidays=None, dfChangepoints=None , ddParams = None):  #get mYhat
        """
        Performs the forecast algorithm at each leaf
        iOoS: of bottom level series
        tune (bool)  : tunes if true

        """ 

        self.dForecasters={}

        if self.type=='temporal': #then it is temporal reconciliation            
            for i,sFreq in enumerate(self.levels):
                data=self.data.resample(sFreq).sum()
                dfY = pd.DataFrame(data=data.values , index=data.index , columns=['y'])
                
                if mX is not None:  #TODO
                    print('No mX')
                else:
                    dfX=None
                    
                pht = Forecast_Prophet(dfData=dfY, dfX=dfX,
                                        dfHolidays=dfHolidays, dfChangepoints=dfChangepoints,
                                        dParams = ddParams[i] if ddParams is not None else None)
                pht.forecast(iOoS=int(time_converter(iOoS,self.levels[-1],sFreq)))
                self.dForecasters[i]=pht
                if sFreq==self.levels[0]:
                    self.mYhat = pht.vYhatOoS.reshape(1,pht.vYhatOoS.shape[0])
                    self.mYhatIS = pht.vYhatIS.reshape(self.dLevels[sFreq], self.mY.shape[1])
                else:
                    mYhat = pht.vYhatOoS.reshape( self.dLevels[sFreq],self.mYhat.shape[1] )
                    mYhatIS = pht.vYhatIS.reshape( self.dLevels[sFreq],self.mY.shape[1] )
                
                    self.mYhatIS = np.vstack((self.mYhatIS,mYhatIS))
                    self.mYhat=np.vstack((self.mYhat,mYhat)) 
        elif self.type=='spatial':
            self.mYhatIS = np.zeros((self.mY.shape[0], self.mY.shape[1]))
            self.mRes = np.zeros((self.mY.shape[0], self.mY.shape[1]))
                        
            n=self.mY.shape[0]
            m=iOoS
            self.mYhat=np.zeros((n,m))
            
            for i in range(self.mY.shape[0]):
                dfY = pd.DataFrame(data=self.mY[i] , index=self.date_time_index , columns=['y'])   
                if mX is not None:
                    dfX = pd.DataFrame(data=mX[i] , 
                                   index=self.date_time_index.append(pd.date_range(start=self.date_time_index[-1] + pd.Timedelta(days=1),
                                                                                                periods=iOoS, freq='D')) , 
                                   columns=['price'])
                else:
                    dfX=None
                            
                pht = Forecast_Prophet(dfData=dfY, dfX=dfX,
                                        dfHolidays=dfHolidays,dfChangepoints=dfChangepoints,
                                        dParams = ddParams[i] if ddParams is not None else None)
                pht.forecast(iOoS=iOoS)
                self.dForecasters[i]=pht
                self.mYhat[i] = pht.vYhatOoS
                self.mYhatIS[i] = pht.vYhatIS                                       
                        
        self.mRes=self.mYhatIS-self.mY    
                         
    def reconcile(self , sWeightType: str):
        """
        Performs whole reconciliation algorithm 
        """                                  
        self.getMatrixW(sWeightType)      
        self.getMatrixP(sWeightType)            
        self.mYtilde=np.dot(np.dot(self.mS,self.mP),self.mYhat)
        
        print('Reconciliation is complete')
    
    def cross_validation(self , dfHolidays, initial, period, horizon , lMethods ):
        """Performs cross_validation and returns matrices required for assesment

        Args:
            initial (_type_): _description_
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
        
        iIters=int((self.mY.shape[1] - initial-horizon)/(period))+1
        print("Number of iterations is " + str(iIters))
 
        dOutputs={}
        
        for method in lMethods:
            dOutputs[method]={}
            
        for iter in range(iIters):   
            tree_iter = copy.copy(self) 
            tree_iter.mY = self.mY[:, 0+period*iter : initial+period*iter ]
            tree_iter.date_time_index=self.date_time_index[ 0+period*iter : initial+period*iter ]
            
            tree_iter_eval = copy.copy(self)
            tree_iter_eval.mY = self.mY[:, period*iter+initial : period*iter+initial+horizon ]  
            if tree_iter_eval.mY.shape[1]!=horizon:
                break    
            
            tree_iter.forecast_Prophet(iOoS=horizon, dfHolidays=dfHolidays, ddParams=self.ddParams)
            tree_iter.forecast_AR()
            
            for sWeightType in lMethods:            
                tree_iter.reconcile(sWeightType)  
                
                if iter!=0:
                    dOutputs[sWeightType]['mYtrue'] = np.hstack([dOutputs[sWeightType]['mYtrue'] , tree_iter_eval.mY[:,-horizon:]])
                    dOutputs[sWeightType]['mYhat'] = np.hstack([dOutputs[sWeightType]['mYhat'] , tree_iter.mYhat[:,-horizon:]])
                    dOutputs[sWeightType]['mYtilde'] = np.hstack([dOutputs[sWeightType]['mYtilde'] , tree_iter.mYtilde[:,-horizon:]])
                else:
                    dOutputs[sWeightType]['mYtrue'] = tree_iter_eval.mY[:,-horizon:]
                    dOutputs[sWeightType]['mYhat'] = tree_iter.mYhat[:,-horizon:]  #TODO is there need for horizon here?
                    dOutputs[sWeightType]['mYtilde'] = tree_iter.mYtilde[:,-horizon:] #TODO is there need for horizon here?
                    dOutputs[sWeightType]['mW']=tree_iter.mW
            print("CV iterations completed = " + str(iter+1) + " of " + str(iIters))
        
        return dOutputs 


        