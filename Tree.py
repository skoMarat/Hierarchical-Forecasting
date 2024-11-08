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
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
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
from forecast_prophet import *
import copy
from datetime import datetime
import pickle


class Tree:
    def __init__(self, data_directory , type: str ):
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
        
        # #reads data
        self.data=pd.read_pickle(data_directory)
        self.type=type
        self.mY , self.date_time_index , self.levels, self.list_of_leafs = self.get_mY()
        
        self.dLevels={}
        for i,l in enumerate(self.levels):
            self.dLevels[self.levels[-i-1]]=len([sublist for sublist in self.list_of_leafs if sublist.count(None) == i]) #70
        self.dLevels['total']=1
        
        self.ddParams = None  #forecasting parameters dictionary of dictionaries   
        self.dForecasters = None # dictionary of forecast instances
        
                   
        self.mP    = None
        self.mW    = None
        self.mYhat = None
        self.mYhatIS = None
        self.mYtilde = None 
        self.mRes = None    # matrix that stores in sample base forecast errors.
        
        # Get summattion matrix S                              
        mS=np.ones((1,next(iter(self.dLevels.values())))) # start with 1 row at the top of matrix s that is always a vector of ones of size equal to # of bottom level series
        
        for i,_ in enumerate(self.levels.to_list()):
            groupByColumns=self.levels.to_list()[:i+1]
            vBtmLevelSeries=self.data.groupby(groupByColumns).count().iloc[:,0].values
            mS=np.vstack([mS,self.create_matrix_S(vBtmLevelSeries)])  
            
        self.mS = mS 

    def get_mY(self):
        """Puts data into a mY according to hierarchical nature
        
        """
        data=self.data
        #based on data, find all possible levels and datetime index
        
        
        def subset_data(data,levels,l:list):
            """
            subsets data to include only data of a certain leaf
            leaf_list (list)  size n, [0] is the level 0 while [-1] is the lowest level
            
            returns: serried of aggregated values for a given leaf_list
            """
            column_mask=(data[levels]==l).any(axis=0)  
            row_mask=(data[levels]==l).loc[:,column_mask].all(axis=1)
            
            srY=data[row_mask].drop(columns=levels).sum(axis=0)
                
            return srY  
        
        if self.type=='spatial':
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
            mY=np.zeros( (len(list_of_leafs), len(date_time_index)))
            
            for i,leaf_creds in enumerate(list_of_leafs):
                mY[i]=subset_data(data, levels,leaf_creds).values                                                              
        elif self.type=='temporal':
        #     #data will be a series not dataframe
        #     levels=['T','H','D','W','M','Q','A']
        #     dLevels={'T': 60 , 'H' :24 , 'D': 7 , 'W':1, 'M': 3 , 'Q':4 , 'A': 1}
            
        #     start_index = levels.index(self.data.index.inferred_freq) #the bottom frequency, the freq of data
        #     end_index = 3 if start_index<3 else 6      
        #     self.levels=levels[start_index:end_index + 1]   
            
        #     #make sure series is summable :  full weeks, full years etc #TODO only works for D W now
        #     start_index = self.data[self.data.index.weekday == 6].index[0]
        #     end_index = self.data[self.data.index.weekday==6].index[-1]
        #     self.data=self.data[start_index:end_index]
            
        #     #create and populate mY
        #     levels=['T','H','D','W','M','Q','A']
        #     dLevels={'T': 60 , 'H' :24 , 'D': 7 , 'W':1, 'M': 3 , 'Q':4 , 'A': 1}

        #     start_index = levels.index(self.data.index.inferred_freq) #the bottom frequency, the freq of data
        #     end_index = 3 if start_index<3 else 6      
        #     levels=levels[start_index:end_index + 1] 

        #     if end_index==3: # works only for W and M data currently
        #         start_index = self.data[self.data.index.weekday == 0].index[0]
        #         end_index = self.data[self.data.index.weekday==6].index[-1]
        #     else:
        #         start_index = self.data[self.data.index.month == 1].index[0]
        #         end_index = self.data[self.data.index.month==12].index[-1]

        #     df=self.data[start_index:end_index]

        #     #create and populate mY
        #     aUnits=np.array([])
        #     n=1
        #     for sFreq in reversed(levels):
        #         n=dLevels[sFreq]*n
        #         aUnits=np.append(aUnits,int(n))
        #     self.aUnits=aUnits[::-1].astype(int)    
        #     n=int(aUnits.sum())
        #     m=df.resample(levels[-1]).sum().shape[0]


        #     mY=df.values.reshape( ( m , aUnits[0] )).T
        #     mS=self.create_matrix([aUnits[0]])
        #     for i,sFreq in enumerate(levels[1:]):
        #         df=df.resample(sFreq).sum()
        #         n = aUnits[i+1]
        #         mY_=df.values.reshape(( m ,n)).T 
        #         mY=np.vstack((mY_, mY))

        #         vBtmLevelSeries=np.full(aUnits[-i-2],int(aUnits[0]/aUnits[-i-2]))
        #         mS_=self.create_matrix(vBtmLevelSeries)
        #         mS=np.vstack(( mS, mS_ ))
            return
        
        return mY, date_time_index , levels, list_of_leafs  

    def create_matrix_S(self, list_of_lengths):
        """
        Creates a matrix with cascading binary entries. 
        list_of_lengths: number of 1's in each row
        number of rows in resulting matrix is equal to number of int in list_of_lengths
         
        """
        total_columns = sum(list_of_lengths)
        matrix = np.zeros((len(list_of_lengths), total_columns), dtype=int)
        
        start_index = 0
        for i, length in enumerate(list_of_lengths):
            matrix[i, start_index:start_index + length] = 1
            start_index += length
        
        return matrix     
                
    def getMatrixW(self , sWeightType:str):
        """
        Purpose:
        create the weights matrix
        
        Outputs:
        mW:            matrix of weights
        """

        mRes=self.mRes.T
        mW = np.eye(mRes.shape[1])
        vNonNanRows = np.setdiff1d(np.arange(0,mRes.shape[0]),  np.unique(np.argwhere(np.isnan(mRes))[:,0]))
        mRes = mRes[vNonNanRows,:]
        
        if sWeightType == 'diag':  #WLS
            for i in range(mRes.shape[1]):
                mW[i,i] = np.mean(mRes[:,i]**2) # error Variance of each leaf
        if sWeightType == 'mint_diag':
            for i in range(mRes.shape[1]):
                mW[i,i] = np.mean(mRes[:,i]**2) # error Variance of each leaf
            mW=np.linalg.inv(mW)      
        elif sWeightType == 'full':  # full
            mSigma = mRes.T @ mRes / mRes.shape[0]
            mW = np.linalg.inv(mSigma)
        elif sWeightType == 'ols':
            mW = np.eye(mRes.shape[1])         
        elif sWeightType == 'mint_shrink':
            n = mRes.shape[0]
            m = mRes.shape[1]
            mWF = mRes.T @ mRes / n
            mWD = np.diag(np.diag(mWF)) # all non-diagonal entries of mWF set to 0
            #calculate numerator
            dBottom = 0 # lower side in the expression for tuning parameter lambda
            for i in range(m):
                for j in range(m):
                    if i>j:
                        dBottom = dBottom + 2*( mWF[i,j] / np.sqrt(mWF[i,i]*mWF[j,j]) )
            #Calculate denominator            
            mResScaled = mRes / np.sqrt(np.diag(mWF)) # elementwise division
            mResScaledSq = mResScaled**2
            mUp = (1/(n*(n-1))) * ( (mResScaledSq.T @ mResScaledSq)- (1/n)*((mResScaled.T @ mResScaled)**2) )
        
            dUp = 0 # lower side in the expression for tuning parameter lambda
            for i in range(m):
                for j in range(m):
                    if i>j:
                        dUp = dUp + 2*mUp[i,j]
            
            dLambda = np.max((np.min((dUp/dBottom, 1)), 0))
            
            # mW = dLambda * np.linalg.inv(mWD) + (1-dLambda) * np.linalg.inv(mWF)
            
            mW = dLambda * mWD + (1-dLambda) * mWF
            mW = np.linalg.inv(mW)         
        # elif sWeightType == 'WRMSSE':
        #     vW=np.loadtxt(os.getcwd()+f"\\data\\M5\\weights.txt")
        #     vW=vW[:self.tree.mY.shape[0]]
        #     vW=vW/vW[0]
        #     mW=np.diag(vW)
        
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
        self.mYhatIS = np.zeros((self.mY.shape[0], self.mY.shape[1]))
        self.mRes = np.zeros((self.mY.shape[0], self.mY.shape[1]))
        self.dForecasters={}

        if self.type=='temporal': #then it is temporal reconciliation
            print('No temporal code available at the moment')
            # vYhat=np.array([])
            # multiple=int(iOoS/self.aUnits[0])
            # for i,sFreq in enumerate(self.levels):  # starts from bottom
            #     dfData=self.data.resample(sFreq).sum()
            #     if sForecMeth=="Prophet":
            #         pht=Forecast_Prophet(dfData=dfData, iOoS=(self.aUnits*multiple)[i])
            #         vYhat_ = pht.forecast(holidays=holidays, changepoints=changepoints).yhat.values
            #     vYhat=np.concatenate((vYhat_, vYhat)) 
            # self.mYhat=vYhat    
             
        else:
            self.mYhat=np.zeros((len(self.list_of_leafs),iOoS))
            
            for i in range(self.mY.shape[0]):
                dfY = pd.DataFrame(data=self.mY[i] , index=self.date_time_index , columns=['y'])   
                if mX is not None:
                    dfX = pd.DataFrame(data=mX[i] , 
                                   index=self.date_time_index.append(pd.date_range(start=self.date_time_index[-1] + pd.Timedelta(days=1),
                                                                                                periods=iOoS, freq='D')) , 
                                   columns=['price'])
                else:
                    dfX=None
                            
                pht = Forecast_Prophet(dfData=dfY, 
                                        dfX=dfX,
                                        dfHolidays=dfHolidays,
                                        dfChangepoints=dfChangepoints,
                                        dParams = ddParams[i] if ddParams is not None else None
                                        )
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
    
    def cross_validation(self , dfHolidays, initial, period, horizon ):
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
        lMethods=["bottom_up", "top_down_ph" ,"top_down_hp",
                  "ols","diag" ,'mint_full','mint_shrink','mint_diag']
        
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
            print("CV iterations completed = " + str(iter+1) + " of " + str(iIters))
            
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
        
        return dOutputs 


        