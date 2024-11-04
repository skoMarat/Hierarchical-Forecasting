import pandas as pd
import numpy as np
import os
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import plotly.graph_objs as go
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
import logging
import cmdstanpy


class Forecast_AR:
    def __init__(self, dfData: pd.DataFrame, dfX=None, dfHolidays=None , dfChangepoints=None , dParams=None):
            """
            dfData   (df)        : dataframe with column y to forecast  and datetime index 
            iOoS (int)           : integer number of oos observations to forecast
            dfX (pd.DataFrame)   : dataframe, same as dfData but with more columns 
            holidays (list)      : list of known holidays or outliers
            changepoints (list)  : list of trend chagepoints
            params (dict)        : dictionary of parameters to be used
            
            """
            self.dfData=dfData    # data , [0] to be forecasted, index must be datetime index
            self.dfX=dfX 
            
            if dfHolidays is not None: # then there are holidays to incorporate
                holiday_dfs=[]
                for i,holiday in enumerate(dfHolidays): 
                    h=pd.DataFrame({
                                    'holiday': str(i+1), #holiday name , #TODO redundant?
                                    'ds': pd.to_datetime([dfHolidays.iloc[i][0]])
                                    })
                    holiday_dfs.append(h)
                    
                #add christmas on top
                mask_is_christmas = np.vectorize(lambda date: (datetime.strptime(date, '%Y-%m-%d').month == 12) & (datetime.strptime(date, '%Y-%m-%d').day == 25))
                christmas=dfHolidays[mask_is_christmas(dfHolidays)]
                if len(christmas)>0:
                    c= pd.DataFrame({
                        'holiday':'Christmas',
                        'ds': pd.to_datetime(christmas.iloc[:,0])
                        })
                    holiday_dfs.append(c)
                    self.dfHolidays=pd.concat(holiday_dfs, ignore_index=True)
            else:
                self.dfHolidays=None
            if dfChangepoints is not None:
                #TODO 
                self.changepoints=None
            else:
                self.changepoints=None
                            
            
            # number of OoS forecast to generate in the same granularity as srY
            self.srYhat=None     # forecasted series TODO
            self.sFreq=self.dfData.index.inferred_freq
            
            if dParams is None:
                #default parameters
                self.changepoint_prior_scale = 0.05 
                self.seasonality_prior_scale = 10
                self.holidays_prior_scale = 10 
                self.seasonality_mode = 'additive'  #LOOK at the data
                self.weekly_seasonality = 3
                self.yearly_seasonality = 10
            else:
                self.changepoint_prior_scale = dParams["changepoint_prior_scale"] 
                self.seasonality_prior_scale = dParams["seasonality_prior_scale"]
                self.holidays_prior_scale = dParams["holidays_prior_scale"] 
                self.seasonality_mode = dParams["seasonality_mode"]
                self.weekly_seasonality = dParams["weekly_seasonality"]
                self.yearly_seasonality = dParams["yearly_seasonality"]
                
            self.model=None
            self.dfModel=None
            self.rmse=None
            self.mape=None
            self.var=None

    def tune():
        return

    def forecast(self):
        return
    
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

        
        
        
    
    
        
            
    
    
