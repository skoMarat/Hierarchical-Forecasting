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
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True
from pandas import to_datetime
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from tqdm import tqdm
import logging
import random
import itertools



"""

This class contains methods for forecasting a single series
Forecast is an object, that must be an object because we might want to generate several forecasts in the future
using that same data and compare. We dont want to run the whole process again just to be able to do that

"""

class Forecast_Prophet:
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
        self.vYhatIS=None
        self.vYhatOoS=None
        self.vRes=None
        
        self.rmse=None
        self.mape=None
        self.var=None
        
        
        
        
    def tune(self, random_size:int , initial:int, period:int, horizon:int , metric:str , parallel='processes' ,plot=False):
        """
        Tunes recommended (by developers) parameters: changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale, seasonality_mode
        
        random_size (int) : size of the random sample from all parameter combinations
        initial     (int) : The initial model will be trained on the first *initial* days of data.
        horizon     (int) : It will forecast the next *horizon* days of data.
        period      (int) : The model will then train on the *initial* days  + *period* days of data and forecast the next *horizon* days.
                            It will continued like this, adding another *period* days to the training data and then forecasting for the next *horizon* days 
                            until there is no longer enough data to do this.
                            Example:
                                1st iteration: train on [i=0 , i=initial] , forecast on [i= initial , i=initial + horizon ]
                                2nd iteration: train on [i=period , i=initial+period] , forecast on [i= initial+period , i=initial+horizon+period]
                                .
                                ..
                                ...
                                nth iteration: .... untial forecast set extends data set
                            
                               
        metric      (str) : Metric for evaluation of fit , choose from: mse, rmse, mae , mape, mdape 
        parallel    (str) : Cross-validation can also be run in parallel mode in Python, by setting specifying the parallel keyword.
                            Choose from: None, processes (not too big problems) , threads, dask (for large, needs installation of DASK)
        plot        (bool): Plots metric used for tuning for diagnostic purposes #TODO
        
        """  
        
        print('Tuning has began')
        
        df=self.dfData['y'].copy().reset_index()
        df.rename(columns={'index' : 'ds'}, inplace=True)
        
        param_grid = {  
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.25 , 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 5 , 10.0],
            'holidays_prior_scale': [0.01 , 0.1, 1 , 5 , 10.0] , 
            'seasonality_mode': ['additive',  'multiplicative'],
            'weekly_seasonality': [3,  7 , 9 ],
            'yearly_seasonality': [5,  10 , 15 , 20]
        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        vMetrics = []  # Store the metrics for each params here
        random_params = random.sample(all_params, min(random_size, len(all_params)))
        #add deafult so that it cant be worse than default
        random_params.append({'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'holidays_prior_scale': self.holidays_prior_scale, 
            'seasonality_mode': self.seasonality_mode,
            'weekly_seasonality': self.weekly_seasonality,
            'yearly_seasonality' : self.yearly_seasonality})
    
        # Use cross validation to evaluate all parameters
        initial = str(initial) + ' days' #TODO link with sFreq
        period  = str(period)  + ' days'
        horizon = str(horizon) + ' days' 
        
        for params in tqdm(random_params, desc='Tuning Progress'):
            m = Prophet(holidays=self.dfHolidays , **params).fit(df)  # Fit model with given params
            df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, parallel=parallel)
            df_p = performance_metrics(df_cv)
            vMetrics.append(df_p[metric].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(random_params)
        tuning_results[metric] = vMetrics
        best_params = random_params[np.argmin(vMetrics)]

        if plot==True: 
            plot_cross_validation_metric(df_cv, metric=metric)
                    
        self.changepoint_prior_scale = best_params['changepoint_prior_scale']
        self.seasonality_prior_scale = best_params['seasonality_prior_scale']
        self.holidays_prior_scale = best_params['holidays_prior_scale'] 
        self.seasonality_mode = best_params['seasonality_mode']  
        self.weekly_seasonality = best_params['weekly_seasonality'] 
        self.yearly_seasonality = best_params['yearly_seasonality'] 
        self.dParams=best_params
        print('Tuning has been terminated succesfully')

        
    def forecast(self, iOoS:int, scaling="absmax") :       # TODO
        """
        Fit Prophet model for daily (or monthly data? TODO)
        regressor self.x must be known also for OoS forecast part. Use other method otherwise
    
    
        
        
        returns nothing , populates self. objects that are required for forecasting
        dfModel.vYhat is the fitted and predicted vYhat of len(self.iOoS + len(dfData))
        
        """
        self.iOoS=iOoS
        if self.sFreq is None:
            print("frequency non inferrable, please provide regular frequency index")
            return None
        
        
        df=self.dfData['y'].copy().reset_index()
        df.rename(columns={'index' : 'ds'}, inplace=True)
                   
        model=Prophet(holidays=self.dfHolidays , 
                      scaling=scaling ,
                      changepoints=self.changepoints,
                      changepoint_prior_scale= self.changepoint_prior_scale,
                      seasonality_prior_scale=self.seasonality_prior_scale ,
                      holidays_prior_scale=self.holidays_prior_scale ,
                      seasonality_mode=self.seasonality_mode,
                      weekly_seasonality=self.weekly_seasonality,
                      yearly_seasonality=self.yearly_seasonality)
        
        if self.dfX is not None: # add regressors
            for regressor in self.dfX.columns:
                #TODO prior scale might need to be adjusted based on regressor 
                model.add_regressor(regressor,prior_scale=0.5, standardize=True)
                df=pd.merge(df,self.dfX[:-iOoS][regressor], how="left", left_on='ds', right_on=self.dfX.index[:-iOoS])
             
        model.fit(df)
        future=model.make_future_dataframe(periods=self.iOoS , freq=self.sFreq)
        
        if self.dfX is not None:
            future[self.dfX.columns]=self.dfX.values
        
        dfModel=model.predict(future)
        
        self.vYhatIS=dfModel.yhat.values[:-self.iOoS]
        self.vYhatOoS=dfModel.yhat.values[-self.iOoS:]
        self.vRes=self.vYhatIS-self.dfData['y']
        self.model=model
        self.dfModel=dfModel
        
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
            


    


        
            
    
    
