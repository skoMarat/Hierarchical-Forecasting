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
        self.dfData=dfData    # data , [0] to be forecasted, index must be datetime index
        self.vY=dfData.values()
        self.date_time_index=dfData.index()
        self.dParams=dParams
                
        self.vYhatOoS=None
        self.vRes=None

        self.model=None
        self.rmse=None
        self.mape=None
        self.var=None      
        
        
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
        
        sFreq=self.date_time_index.inferred_freq
        if sFreq=='D':
            p_range=7 , d_range=2 , q_range=7 
        elif sFreq=='W':
            p_range=7 , d_range=2 , q_range=7

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
        
        forecast=self.model.get_forecast(steps=iOoS).predicted_mean
        fit=self.model.fittedvalues
        
        self.vYhatIS=fit.values[iOoS:]
        self.vYhatOoS=forecast.values
        self.vRes=self.vYhatIS-self.vY[iOoS:]
        
        #populate performance metrics
        self.rmse=np.sqrt(np.mean((self.vYhatIS-self.dfData.y )** 2))
        self.mape=np.mean(np.abs((self.dfData.y[self.dfData.y != 0] - self.vYhatIS[self.dfData.y != 0]) / self.dfData.y[self.dfData.y != 0])) * 100
        self.var=(self.vYhatIS-self.dfData.y ).var()
    
    
        
        
    def plot_prediction(self,inSample=True):
        """
        Plots prediction and data, also insample prediction if bool=True 
        """    
        srYhatIS=pd.Series(self.vYhatIS,index=self.dfData.index)
        
        if self.sFreq == 'M':
            start = self.dfData.index[-1] + pd.DateOffset(months=1)
        elif self.sFreq == 'D':
            start = self.dfData.index[-1] + pd.DateOffset(days=1)
        elif self.sFreq == 'W' or self.sFreq == 'W-SUN' :
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
        
        # Plot errors
        plt.figure(figsize=(12, 6))
        if inSample==True:
            plt.plot(srYhatIS.index , srYhatIS - self.dfData.y, label='Residual', color='red')
        else:
            plt.plot(srYhatIS.index[-self.iOoS*2:] , srYhatIS[-self.iOoS*2:] - self.dfData[-self.iOoS*2:].y, label='Residual', color='red')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Residuals time-series')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(12, 6))
        if inSample==True:
            plt.hist( srYhatIS - self.dfData.y, label='Residuals')
        else:
            plt.hist( srYhatIS[-self.iOoS*2:] - self.dfData[-self.iOoS*2:].y, label='Residuals')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Residuals histogram')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
        #residual autocorrelation
        plot_acf(srYhatIS - self.dfData.y, lags=31)  
        plt.xlabel('Lags')
        plt.title('Autocorrelation Function (ACF) of Residuals') 
        plt.ylabel('Autocorrelation')
        plt.show()     
        
        
        fig = self.model.plot_components(self.dfModel)
        fig
            


    


        
            
    
    
