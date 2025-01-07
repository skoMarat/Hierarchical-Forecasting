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
from scipy.stats import probplot
from dask.distributed import Client



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
                                   changpoint_prior_scale: flexibility of trend, small values leads to underfit of trend
                                   seasonality_prior_scale: flexibility of seasons, large value allows to fit large fluctuations
                                   holidays_prior_scale : flexibility of holidays, higher values offer no regularization
                                   seasonality_mode : if the magniture of seasonal fluctuations grows with magnitude of the time series then multiplicative
        
        """
        self.dfData=dfData    # data , [0] to be forecasted, index must be datetime index
        self.dfX=dfX 
        self.sTransform=None
        self.sFreq=pd.infer_freq(dfData.index)[0] # get W of W-SUN
        
        if dfHolidays is not None: # then there are holidays to incorporate
            holiday_dfs=[]
            for i in range(dfHolidays.shape[0]):  
                if self.sFreq=='D':
                    h=pd.DataFrame({
                                    'holiday': str(i+1), 
                                    'ds': pd.to_datetime([dfHolidays.iloc[i][0]])
                                    })
                    holiday_dfs.append(h)
                elif self.sFreq=='W':
                    holiday_date = pd.to_datetime(dfHolidays.iloc[i][0])
                    sunday_date = (holiday_date + timedelta(days=(6 - holiday_date.weekday()))).strftime('%Y-%m-%d')
                    h=pd.DataFrame({
                                    'holiday': str(i+1), #holiday name , #TODO redundant?
                                    'ds': pd.to_datetime([sunday_date])
                                    })
                    holiday_dfs.append(h)
                
            #add christmas on top
            # mask_is_christmas = np.vectorize(lambda date: (datetime.strptime(date, '%Y-%m-%d').month == 12) & (datetime.strptime(date, '%Y-%m-%d').day == 25))
            # christmas=dfHolidays[mask_is_christmas(dfHolidays)]
            # if len(christmas)>0:
            #     c= pd.DataFrame({
            #         'holiday':'Christmas',
            #         'ds': pd.to_datetime(christmas.iloc[:,0])
            #         })
            #     holiday_dfs.append(c)
            #     self.dfHolidays=pd.concat(holiday_dfs, ignore_index=True)
            self.dfHolidays=pd.concat(holiday_dfs, ignore_index=True)
        else:
            self.dfHolidays=None
        if dfChangepoints is not None:
           #TODO 
           self.changepoints=None
        else:
            self.changepoints=None
                           
        
        if dParams is None:
            #default parameters
            self.changepoint_prior_scale = 0.05 
            self.seasonality_prior_scale = 10
            self.holidays_prior_scale = 10 
            self.seasonality_mode = 'additive'  #LOOK at the data
            self.weekly_prior_scale =10
        else:
            self.changepoint_prior_scale = dParams["changepoint_prior_scale"] 
            self.seasonality_prior_scale = dParams["seasonality_prior_scale"]
            self.holidays_prior_scale = dParams["holidays_prior_scale"] 
            self.seasonality_mode = dParams["seasonality_mode"]
            # self.weekly_seasonality = dParams["weekly_seasonality"]
            # self.monthly_seasonality = dParams["monthly_seasonality"]
            # self.yearly_seasonality = dParams["weekly_seasonality"]
       
            
        self.model=None
        self.dfModel=None
        self.vYhatIS=None
        self.vYhatOoS=None
        self.vRes=None
        
        self.rmse=None
        self.mape=None
        self.var=None
     
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
            
    def tune(self, iSize:int , iInitial:int, iPeriod:int, iHorizon:int , sMetric:str , parallel='processes' , bPlot=False):
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
            'changepoint_prior_scale': [0.001,  0.1,  0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 5 ,  10.0],
            'holidays_prior_scale': [0.01 , 1 ,  10.0] , 
            'seasonality_mode': ['additive',  'multiplicative'],
            'weekly_prior_scale': [ 0.01,  0.1 , 1 ,  5 , 10 ],
            # 'monhtly_seasonality': [ 3 , 4, 5 , 6] ,
            # 'yearly_seasonality': [ 10 , 15 , 20 , 25 ]
        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        vMetrics = []  # Store the metrics for each params here
        random_params = random.sample(all_params, min(iSize, len(all_params)))
        #add deafult so that it cant be worse than default
        random_params.append({
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'holidays_prior_scale': self.holidays_prior_scale, 
            'seasonality_mode': self.seasonality_mode,
            'weekly_prior_scale': self.weekly_prior_scale
            })
    
        # Use cross validation to evaluate all parameters
        initial = str(iInitial) + " "+self.sFreq 
        period  = str(iPeriod)  + " "+self.sFreq 
        horizon = str(iHorizon) + " "+self.sFreq  
        
        for params in tqdm(random_params, desc='Tuning Progress'):
            m = Prophet(holidays=self.dfHolidays , changepoint_prior_scale=params['changepoint_prior_scale'],
                        seasonality_prior_scale=params['seasonality_prior_scale'],
                        holidays_prior_scale= params['holidays_prior_scale'],
                        seasonality_mode=params['seasonality_mode']
                        )  # Fit model with given params
            if self.sFreq=='D':
                m.add_seasonality(name='weekly', 
                                  period=7, fourier_order=3, 
                                  prior_scale=params['weekly_prior_scale'])
            if self.sFreq=='W':
                m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            m.add_seasonality(name='yearly', period=365.5, fourier_order=10)
            m.fit(df)
            
            df_cv = cross_validation(m, initial=initial, period=period,
                                     horizon=horizon, parallel=parallel)
            df_p = performance_metrics(df_cv)
            vMetrics.append(df_p[sMetric].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(random_params)
        tuning_results[sMetric] = vMetrics
        best_params = random_params[np.argmin(vMetrics)]

        if bPlot==True: 
            plot_cross_validation_metric(df_cv, metric=sMetric)
                    
        self.changepoint_prior_scale = best_params['changepoint_prior_scale']
        self.seasonality_prior_scale = best_params['seasonality_prior_scale']
        self.holidays_prior_scale = best_params['holidays_prior_scale'] 
        self.seasonality_mode = best_params['seasonality_mode']  
        self.weekly_prior_scale = best_params['weekly_prior_scale']

        self.dParams=best_params
        print('Tuning has been terminated succesfully')

        
    def forecast(self, iOoS:int, scaling="absmax") :       
        """
        Fit Prophet model for daily (and weekly data? TODO)
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
                      seasonality_mode=self.seasonality_mode)
        
        if self.sFreq=='D':
            model.add_seasonality(name='weekly', period=7, fourier_order=5 , 
                                  prior_scale= 20 )
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        if self.sFreq=='W':
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='yearly', period=365.5, fourier_order=10)
        
        
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
        self.vRes=self.vYhatIS-self.dfData['y'].values
        self.model=model
        self.dfModel=dfModel
        
        #populate performance metrics
        self.rmse=np.sqrt(np.median((self.vYhatIS-self.dfData.y )** 2))
        self.mape=np.median(np.abs((self.dfData.y[self.dfData.y != 0] - self.vYhatIS[self.dfData.y != 0]) / self.dfData.y[self.dfData.y != 0])) * 100
        self.var=(self.vYhatIS-self.dfData.y ).var()
        
        self.retransform()
           
        
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


    


        
            
    
    
