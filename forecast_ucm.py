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
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import probplot
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.sarimax import SARIMAX



"""

This class contains methods for forecasting a single series
Forecast is an object, that must be an object because we might want to generate several forecasts in the future
using that same data and compare. We dont want to run the whole process again just to be able to do that

"""

class Forecast_UCM:
    def __init__(self, dfData: pd.DataFrame , sModel='lltrend' ):
        """
        dfData   (df)        : dataframe with column y to forecast, with NaNs for values that needs to be forecasted
                                column(s) exogs for past and future values of exog variables
        iOoS (int)           : integer number of oos observations to forecast

        
        """
        self.dfData = dfData    # data , [0] to be forecasted, index must be datetime index
        self.vY = dfData.y.dropna().values
        self.srY=self.dfData.y[:len(self.vY)]      
        self.exog_vars=self.dfData[['price','christmas',
                        'snap','holidays'
                        ]][:len(self.vY)]
        self.exog_vars_future=dfData[['price','christmas',
                        'snap','holidays'
                        ]][len(self.vY):]

        self.sModel=sModel
        self.sFreq = dfData.index.inferred_freq                
        self.vYhatOoS = None
        self.iOoS     = None
        self.vRes     = None

        self.model = None 
        self.rmse  = None
        self.mape  = None
        self.var   = None      
 
    
        
    def fit(self):
        """
        fit model by finding the parameters using MLE

        Parameters:

        Attributes:
        - self.dParams: Dictionary of optimal parameters 
        - self.model: fitted model.
        """
        if self.sFreq == 'D':
            freq_seasonal = [
                {'period': 7, 'harmonics': 3},
                {'period': 30, 'harmonics': 4},
                {'period': 365, 'harmonics': 5} # should probably be higher
            ]
            # stochastic_freq_seasonal=[True, True, True]
            # autoregressive=7

        elif 'W' in self.sFreq:
            freq_seasonal = [
                {'period': 52, 'harmonics': 5},
                {'period': 4, 'harmonics': 3}
            ]  
            autoregressive=5
            # stochastic_freq_seasonal=[True, True]
            
         
        try:    
            self.model=UnobservedComponents(self.vY, self.sModel,
                        freq_seasonal=freq_seasonal,
                        # stochastic_freq_seasonal=stochastic_freq_seasonal,
                        exog=self.exog_vars,
                        # irregular=True,
                        # autoregressive=autoregressive,
                        initialization='approximate_diffuse'
                        ).fit(low_memory=True)
        except:
            print('Non AR model fitted')
            self.model=UnobservedComponents(self.vY, self.sModel,
                        freq_seasonal= [
                                        {'period': 7, 'harmonics': 3},
                                        {'period': 30, 'harmonics': 4},
                                        {'period': 365, 'harmonics': 5} # should probably be higher
                                    ],
                        stochastic_freq_seasonal=[True, True, True],
                        exog=self.exog_vars,
                        irregular=False,
                        ).fit(low_memory=True)
            
  



    def transform(self, sType:str):
        """
        Transforms the data 
        Apply before doing any tuning or forecasting
        """    
        self.sTransform=sType
        
        if self.sTransform=='log':
            self.vY=np.log(self.vY + 1e-6)
            self.exog_vars['price']=np.log( self.exog_vars['price'] + 1e-6)
            self.exog_vars_future['price']=np.log( self.exog_vars_future['price'] + 1e-6)
            if self.vYhatOoS is not None:

                self.vYhatIS=np.exp(self.vYhatIS)
                self.vYhatOoS=np.exp(self.vYhatOoS)
                self.srYhatIS=np.exp(self.srYhatIS)
                self.srYhatOoS=np.exp(self.srYhatOoS)
                
                self.vRes=self.vYhatIS-self.vY
                self.vRes=self.vRes[7*6:]
                
                #populate performance metrics
                self.rmse=np.sqrt(np.median((self.vYhatIS-self.vY )** 2))
                self.mape=np.median(np.abs((self.vY[self.vY != 0] - self.vYhatIS[self.vY != 0]) / self.vY[self.vY != 0])) * 100
                self.var=(self.vYhatIS-self.vY ).var() 

                
                
    def retransform(self):
        """
        Transform back to original scale
        Perform after tuning and forecasting
        """
        if self.sTransform=='log':
            self.vY=np.exp(self.vY)
            
            if self.vYhatOoS is not None:
                self.vYhatIS=np.exp(self.vYhatIS)
                self.vYhatOoS=np.exp(self.vYhatOoS)
                self.srYhatIS=np.exp(self.srYhatIS)
                self.srYhatOoS=np.exp(self.srYhatOoS)
                
                self.vRes=self.vYhatIS-self.vY
                self.vRes=self.vRes[7*6:]
                
                #populate performance metrics
                self.rmse=np.sqrt(np.median((self.vYhatIS-self.vY )** 2))
                self.mape=np.median(np.abs((self.vY[self.vY != 0] - self.vYhatIS[self.vY != 0]) / self.vY[self.vY != 0])) * 100
                self.var=(self.vYhatIS-self.vY ).var()
        else:
            return
        
        
    def forecast(self, iOoS:int) :       
        """_summary_

        Args:
            iOoS (int): _description_

        Returns:
            _type_: _description_
        """
        self.iOoS = iOoS

        forecast=self.model.get_forecast(steps=iOoS, exog=self.exog_vars_future[:iOoS])
        self.vYhatIS=self.model.fittedvalues.values
        self.vYhatOoS=forecast.predicted_mean.values
        
        self.srYhatIS  = pd.Series( self.vYhatIS  ,    index=self.dfData.index[:len(self.vYhatIS)])
        self.srYhatOoS = pd.Series( self.vYhatOoS ,    index=self.dfData.index[len(self.vY):][:len(self.vYhatOoS)])
        
        
        self.vRes=self.vYhatIS-self.vY
        self.vRes=self.vRes[7*6:]
        
        #populate performance metrics
        self.rmse=np.sqrt(np.mean((self.vYhatIS-self.vY )** 2))
        # self.mape=np.mean(np.abs((self.vY[self.vY != 0][iOoS:] - self.vYhatIS[self.vY != 0]) / self.vY[self.vY != 0][iOoS:])) * 100
        self.var=(self.vYhatIS-self.vY ).var()
       
    def plot_prediction(self):
        """
        Plots prediction and data, also insample prediction if bool=True 
        """    
          
        plt.figure(figsize=(12, 6))
        # Plot actuals
        if self.sFreq== 'D':
            plt.plot(self.srY[-self.iOoS*10:] , label='Actual Data', color='green',linestyle='', marker='o',markersize=4)
            plt.plot(self.srYhatIS[-self.iOoS*10:] , label='In-Sample Forecast', color='blue', linestyle='--')
            plt.plot(self.srYhatOoS, label='Out-of-Sample Forecast', marker='o', color='red', linestyle=':')  
        else:    
            plt.plot(self.srY , label='Actual Data', color='green',linestyle='', marker='o',markersize=4)
            plt.plot(self.srYhatIS , label='In-Sample Forecast', color='blue', linestyle='--')
            plt.plot(self.srYhatOoS, label='Out-of-Sample Forecast', marker='o', color='red', linestyle=':')
        for i in range(10):
            plt.axvline(x=self.srY.index[-7*i-1], color='black', linestyle='--', alpha=0.2) 
        # Customize the plot
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('In-Sample, and Out-of-Sample Forecasts')
        plt.legend()
        plt.grid(True)
        plt.show()      
        
        ##################
        # fig = self.model.plot_components(figsize=(12, 12))
        # plt.tight_layout()  # Ensure proper spacing
        # plt.show()
        
    def print_summary(self):
        print(self.model.summary())    
            
    def residual_diagnostic(self):
        """
        Plots residual diagnostics 
        """    
        # Plot errors
        plt.figure(figsize=(12, 6))
        plt.plot(self.srYhatIS.index[-len(self.vRes):] , self.vRes, label='Residual', color='red')
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


    
    


        
            
    
    
