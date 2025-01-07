from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
from numpy.linalg import inv
import os
from datetime import datetime,timedelta
import re
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
import warnings
import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True
from pandas import to_datetime
from tqdm import tqdm
import logging
import random
import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller



"""

This class contains methods for forecasting a single series
Forecast is an object, that must be an object because we might want to generate several forecasts in the future
using that same data and compare. We dont want to run the whole process again just to be able to do that

"""

class Forecast_ARIMA:
    def __init__(self, dfData: pd.DataFrame , dParams=None):
        """
        dfData   (df)        : dataframe with column y to forecast  and datetime index 
        iOoS (int)           : integer number of oos observations to forecast

        
        """
        self.dfData = dfData    # data , [0] to be forecasted, index must be datetime index
        self.vY = dfData.values
        self.date_time_index = dfData.index
        self.sFreq = dfData.index.inferred_freq
        self.dParams = dParams
                
        self.vYhatOoS = None
        self.iOoS     = None
        self.vRes     = None

        self.model = None 
        self.rmse  = None
        self.mape  = None
        self.var   = None      
        
        
    def tune(self):
        """
        Tune ARIMA(p, d, q) model by finding the best combination of p, d, and q 
        that minimizes the AIC value.

        Parameters:
        - p_range: int: defined the upper Range of AR values to test.
        - d_range: int: defined the upper Range of differencing values to test.
        - q_range: int: defined the upper Range of MA values to test.

        Attributes:
        - self.dParams: Dictionary of optimal parameters {'p': best_p, 'd': best_d, 'q': best_q}.
        - self.model: The best ARIMA model.
        """
        best_aic = float('inf')
        best_order = None 
        best_model = None  
        
        if self.sFreq=='D':
            p_range=7 
            d_range=2
            q_range=7 
        elif self.sFreq=='W':
            p_range=5
            d_range=2
            q_range=5
        else:
            print('unspecified freq')
   
        # Grid search over p, d, and q
        for p in range(0,p_range):
            for d in range(0,d_range+1):
                for q in range(0,q_range+1):
                    try:
                        # Fit ARIMA model with (p, d, q)
                        model = ARIMA(self.vY, order=(p, d, q)).fit()
                        
                        # Check if the current model has a better AIC
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, d, q)
                            best_model = model
                    except Exception as e:
                        continue  # Skip invalid combinations

        # Save best parameters and model
        if best_order:
            self.dParams = {'p': best_order[0], 'd': best_order[1], 'q': best_order[2]}
            self.model = best_model
            print('Tuning has been terminated successfully.')
        else:
            print('No valid model found. Please check your input ranges or data.')


    def transform(self, sType:str):
        """
        Transforms the data 
        Apply before doing any tuning or forecasting
        """    
        self.sTransform=sType
        
        if self.sTransform=='log':
            self.dfData=np.log(self.dfData + 1e-6)
            if self.dfX is not None:
                self.dfX=np.log(self.dfX + 1e-6) 
                
    def retransform(self):
        """
        Transform back to original scale
        Perform after tuning and forecasting
        """
        if self.sTransform=='log':
            self.dfData=np.exp(self.dfData)
            self.vYhatIS=np.exp(self.vYhatIS)
            self.vYhatOoS=np.exp(self.vYhatOoS)
            # self.vRes=self.vYhatIS-self.dfData['y'].values
            
            #populate performance metrics
            self.rmse=np.sqrt(np.median((self.vYhatIS-self.dfData.y )** 2))
            self.mape=np.median(np.abs((self.dfData.y[self.dfData.y != 0] - self.vYhatIS[self.dfData.y != 0]) / self.dfData.y[self.dfData.y != 0])) * 100
            self.var=(self.vYhatIS-self.dfData.y ).var()
        else:
            return
        
        
    def perform_ADF(self):
        """
        Perform Augment Dickey Fuller test for stationarity
        """   
        result = adfuller(self.vY)
        if result[1] <= 0.05:
            return 'stationary'
        else:
            return 'non-stationary' 
        
    def forecast(self, iOoS:int) :       
        """_summary_

        Args:
            iOoS (int): _description_

        Returns:
            _type_: _description_
        """
        self.iOoS = iOoS
        
        p=self.dParams['p']
        d=self.dParams['d']
        q=self.dParams['q']
        
        self.model = ARIMA(self.vY, order=(p, d, q)).fit()
        forecast=self.model.get_forecast(steps=iOoS).predicted_mean
        fit=self.model.fittedvalues
        
        self.vYhatIS=fit
        self.vYhatOoS=forecast
        self.vRes=self.vYhatIS-self.vY
        
        #populate performance metrics
        self.rmse=np.sqrt(np.mean((self.vYhatIS-self.vY )** 2))
        # self.mape=np.mean(np.abs((self.vY[self.vY != 0][iOoS:] - self.vYhatIS[self.vY != 0]) / self.vY[self.vY != 0][iOoS:])) * 100
        self.var=(self.vYhatIS-self.vY ).var()
    
    def plot_prediction(self,inSample=True):
        """
        Plots prediction and data, also insample prediction if bool=True 
        """    
        srYhatIS=pd.Series(self.vYhatIS,index=self.dfData.index)
        
        if self.sFreq == 'M':
            start = self.dfData.index[-1] + pd.DateOffset(months=1)
        elif self.sFreq == 'D':
            start = self.dfData.index[-1] + pd.DateOffset(days=1)
        elif self.sFreq == 'W' :
            start = self.dfData.index[-1] + pd.DateOffset(weeks=1)
        elif self.sFreq == 'H':
            start = self.dfData.index[-1] + pd.DateOffset(hours=1)
        elif self.sFreq == 'Q':
            start = self.dfData.index[-1] + pd.DateOffset(months=3)
        elif self.sFreq == 'Y':
            start = self.dfData.index[-1] + pd.DateOffset(years=1)
        else:
            raise ValueError(f"Unsupported frequency: {self.sFreq}")
        periods=self.iOoS
        srYhatOoS=pd.Series(self.vYhatOoS,index=pd.date_range(start=start,periods=periods,freq=self.sFreq) )          
        
        plt.figure(figsize=(12, 6))
        if inSample==True:
            # Plot dfData
            plt.plot(self.dfData.index, self.dfData, label='Actual Data', color='green',linestyle='', marker='o',markersize=4)
            # Plot vYhatIS
            plt.plot(srYhatIS.index, srYhatIS, label='In-Sample Forecast', color='blue', linestyle='--')
            
        else:
            # Plot dfData
            plt.plot(self.dfData.index[-self.iOoS*2:], self.dfData[-self.iOoS*2:], label='Actual Data', color='green',linestyle='', marker='o',markersize=4)
            # Plot vYhatIS
            plt.plot(srYhatIS.index[-self.iOoS*2:], srYhatIS[-self.iOoS*2:], label='In-Sample Forecast', color='blue', linestyle='--')
            
            
        # Plot vYhatOoS
        plt.plot(srYhatOoS.index, srYhatOoS, label='Out-of-Sample Forecast', color='red', linestyle=':')
        # Customize the plot
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('In-Sample, and Out-of-Sample Forecasts')
        plt.legend()
        plt.grid(True)
        plt.show()      
        
        fig = self.model.plot_components(self.dfModel)
        fig
            
    def residual_diagnostic(self):
        """
        Plots residual diagnostics 
        """    
        # Plot errors
        plt.figure(figsize=(12, 6))
        plt.plot(self.dfData.index , self.vRes, label='Residual', color='red')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Residuals time-series')
        plt.legend()
        plt.grid(True)
        plt.show() 
        
        plt.hist( self.vRes, label='Residuals')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Residuals histogram')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(12, 6))
        probplot(self.vRes, dist="norm", plot=plt)
        plt.title('Residuals QQ Plot')
        plt.grid(True)
        plt.show()
                
        #residual autocorrelation
        plot_acf(self.vRes, lags=31)  
        plt.xlabel('Lags')
        plt.title('Autocorrelation Function (ACF) of Residuals') 
        plt.ylabel('Autocorrelation')
        plt.show()    
        
        plot_pacf(self.vRes, lags=31)  
        plt.xlabel('Lags')
        plt.title('Partial Autocorrelation Function (PACF) of Residuals') 
        plt.ylabel('Partial Autocorrelation')
        plt.show()  


    
    


        
            
    
    
