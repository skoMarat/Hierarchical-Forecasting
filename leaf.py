from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
from numpy.linalg import inv
import os
import pandas as pd
import numpy as np
import os
from datetime import datetime,timedelta
from scipy.interpolate import splrep, splev
import re
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly.graph_objs as go
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
import logging
import cmdstanpy
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
from dateutil.relativedelta import relativedelta
import  skfda
from skfda.preprocessing.dim_reduction import FPCA
import fdata
from fdata import FData



class Leaf:
    def __init__(self, date:str ,  data:str, *levels ):
        """ 
        A Leaf object represents each curve data in the hierarhcy
     
        Thus not specifying any L2,L3,L4 results in creation of leaf L1 -> AFKL
             not specifying L3 but L2, results in creating a leaf for specified L2
             not specifying L2 but L3 is illegal
        Args:
            date (str):          3 month ahead date to be forecasted,
                                begining of the month date
                                Example: if we are at second week of september
                                and we want forecast for October, November, December
                                We should insert date '2023-12-01'
            args (str)          Level , possibly more than 2
        """
        
        for i, level in enumerate(levels):
            setattr(self, f"Level_{i}", level)     
        # self.L2=L2
        # self.L3=L3
        # self.L4=L4
        
        # self.data_location=data_location
        self.date=date
        
        self.raw_data=self.get_smooth_data()  # gets all raw_data also oos if in training mode
        self.data=self.raw_data.copy().loc[:, :date]   # all and only fully observed curves
        # TODO in future, this is fully and partially observed curves only
        self.vY3=self.raw_data[pd.to_datetime(date) - relativedelta(months=0)].copy() 
        self.vY2=self.raw_data[pd.to_datetime(date) - relativedelta(months=1)].copy()
        self.vY1=self.raw_data[pd.to_datetime(date) - relativedelta(months=2)].copy()
        self.vY0=self.raw_data[pd.to_datetime(date) - relativedelta(months=3)].copy()

        self.vY0E=self.vY0 [:-(4*0+2)]  #partially observed
        self.vY1E=self.vY1 [:-(4*1+2)]  #partially observed
        self.vY2E=self.vY2 [:-(4*2+2)]  #partially observed
        self.vY3E=self.vY3 [:-(4*3+2)]  #partially observed
        self.list_vYE=[self.vY0E,self.vY1E,self.vY2E,self.vY3E] 

        #save actual valus vectors , for training
        self.vY0L=self.vY0.copy() [-(4*0+2):]  #unobserved, =nan in production
        self.vY1L=self.vY1.copy() [-(4*1+2):]  #unobserved
        self.vY2L=self.vY2.copy() [-(4*2+2):]  #unobserved
        self.vY3L=self.vY3.copy() [-(4*3+2):]  #unobserved
        self.list_vYL=[self.vY0L,self.vY1L,self.vY2L,self.vY3L]  #keeping this block now for training purposes
        
        #reconciled forecast, populated at the end of algo in reconciliation class
        self.vY0L_rec= None #self.vY0L.copy()  #unobserved, =nan in production
        self.vY1L_rec= None #self.vY1L.copy()
        self.vY2L_rec= None #self.vY2L.copy()  #unobserved
        self.vY3L_rec= None #self.vY3L.copy()  #unobserved     
        self.list_vYL_rec=[[]]*4
             
        # in future, it is assumed that these are already the case TODO
        self.vY0[-(4*0+2):] = np.nan
        self.vY1[-(4*1+2):] = np.nan
        self.vY2[-(4*2+2):] = np.nan  
        self.vY3[-(4*3+2):] = np.nan
        self.list_vY = [self.vY0,self.vY1,self.vY2,self.vY3] #usefull for clean plotting and appending to data during algorithm execution
        #####
        
        self.list_vYhat=[[]]*4 # no predicted observation during initialization
        self.list_vYL_rec=[[]]*4
        self.data = self.data.iloc[:,:-4]   #only fully observed curves in the begining of algorithm
        
        self.vErrorIS=None #will store I.S. error vectors to be used for reconciliation
        
        #functional data variables
        self.mBhat= None
        self.list_fdataY=[[]] * 4

        if (self.L2 is None) and (self.L3 is None) and (self.L4 is None):
            print(f"Created a leaf for AFKL")
        elif (self.L3 is None) and (self.L4 is None):
            print(f"Created a leaf for {self.L2}")
        elif (self.L4 is None):
            print(f"Created a leaf for {self.L2} {self.L3}")
        elif (self.L2 is not None) and (self.L3 is not None) and (self.L4 is not None):
            print(f"Created a leaf for {self.L2} {self.L3} {self.L4}")
        
        
    def get_smooth_data(self):
        """Gets smooth data from lowest level hierarchy files in data folder
        For higher level hierarchies? 
   
        Returns:
            _type_: _description_
        """
        path=os.getcwd()
        if self.L4 is None: 
            listL4=['KL','AF']
        else:
            listL4=[self.L4]
        if self.L3 is None:
            listL3=['NL' , 'BE' , 'LU']
        else:
            listL3=[self.L3]
        if self.L2 is None:
            listL2=['Africa','Asia','France','North America','Caribbean & Indian Ocean',
            'Central & South America','Europe and North Africa','Middle East' ] 
        else:
            listL2=[self.L2]
            
        # raw data is from Level4, for higher levels, it needs to be summed 
        date_range = pd.date_range(start='2019-01-01', end=self.date, freq='MS')  #same date as smooth_data[-1]
        accumulated_raw_data = pd.DataFrame(0, index=range(0,42), columns=date_range)

        for L4 in listL4:
            for L3 in listL3:
                for L2 in listL2:
                    if L4=='KL' and L2=='France':
                        continue  #since it does not exist

                    ###########################################################################
                    if self.data_location is None:
                        smooth_data=pd.read_csv(path+f"\\data\\{self.KPI}_data_cumulative\\{L4}_{L3}_{L2}.csv")
                    else: 
                        smooth_data=pd.read_csv(path+self.data_location)

                    #smooth_data=smooth_data.set_index(np.arange(0,42)[::-1])
                    # raw_data=raw_data.loc[::-1]
                    smooth_data.columns = pd.to_datetime(smooth_data.columns)
                    # raw_data = raw_data.sort_index(axis=1)
                    # raw_data=raw_data.bfill() # Replace NaN with value that comes below it in the same column
                    # data=raw_data_sorted.copy()
                    ###############################################################################
                    accumulated_raw_data=accumulated_raw_data.add(smooth_data)
        smooth_data=accumulated_raw_data
                
        return smooth_data
    
    
    def forecast_betas_VAR(self , mBetaE, mBeta):
        """
        Fits a regression line for provided data and 
        returns matrix length ammount of one step ahead forecast with VAR
        inputs :
        
        
        """
        for i in mBetaE.shape[1]:
            return    
               
            
    def forecast_betas_prophet(self, dfmBeta, dfmBetaE, m = 1, changepoints = True , 
                    holidays =False,  scale='absmax' ,
                    changepoint_prior_scale=0.05 , holidays_prior_scale=0.9,
                    yearly_seasonality=20):
        """
        Using Prophet, predicts m steps ahead for each PC given in mBeta
        Args:
            mBeta (np array): matrix of betas of fully realized curves
            m : int , number of months(steps) ahead to predict, default=1 
            ind: datetime index of mBeta
            changepoints (bool): if True, use 2022-03-01 as a changepoint (read break) for trend , defaults to True
            holidays (bool):     if True, used 2022 as a transition period from a shock and discount error and values effect on trend, defaults to True
            scale (str):         'absmax' or 'minmax'
            holidays_prior_scale (int): defaults to 10 ,  which provides very little regularization. 
                                 If you find that the holidays are overfitting, you can adjust their prior scale to smooth them. 
                                 Reducing this parameter dampens holiday effects:
            changepoint_prior_scale (float): defaults to 0.05, increasing will lead to more flexibile trend component
            yearly_seasonality (int):   defaults to 20 , number of Fourier series fitted for seasonality, 
                                 increase will fill more series and thus lead to more complex seasonality and higher flexibility of seasonal component
   

            dfmBetaE (pd df) : must includes future values 
            
        Outputs:
           
           vBhat: Out of sample beta forecasts
           df: df used in forecasting
           mBhatIS: in sample betas    

        """

        K=dfmBeta.shape[1]
        vBhat=np.zeros(K)
        mBhatIS=np.zeros((len(dfmBeta),K))   #n by k matrix, n=number of insample time indeces= len(dfmBeta), k is number of iPC
        for i in range(K):
            df=dfmBeta[i]
            df=df.reset_index()
            df.columns=['ds','y']
    
            if dfmBetaE is not None:
                #Add regressors        
                df=pd.merge(df,dfmBetaE.iloc[:-m,i],how='left', left_on='ds',right_on=dfmBetaE.index[:-m])
                #so only merging for now the non forecast part
                df.columns=['ds','y','mBetaE']    
                        
            #define model    
            model=Prophet(
                        scaling=scale,   
                        changepoint_prior_scale=changepoint_prior_scale,
                        holidays_prior_scale=holidays_prior_scale,
                        yearly_seasonality=yearly_seasonality
                    )        
            
            if dfmBetaE is not None:
                model.add_regressor('mBetaE', prior_scale=0.5, standardize='True')    

            model.fit(df)
            
            future = model.make_future_dataframe(periods=m, freq='MS')
            if dfmBetaE is not None:
                future['mBetaE']=dfmBetaE.loc[:,i].values
                
            dfModel=model.predict(future)
            vBhat[i]=dfModel.yhat.iloc[-1]
            mBhatIS[:,i]=dfModel.yhat[:-1] 
               
        return vBhat , mBhatIS

    def forecast_leaf( self, sForecMeth ,  iPC=3, n_basis=10 ):
        """
        Forecast a given hierarchy, for a given date, KPI, using parameters for FDA and Forecasting
        
            subset (int)         Number of weeks in the begining of the curve to ignore, defualts to 0, setting to high can lead to error in pc extraction part
            iPC (int) :          Number of principal components used in decomposing of the curve, defaults to 3
            n_basis (int) :      Number of basis functions to use in smoothing of the curve, defaults to 10         
        Output: 
            mvYhat : 4*p of forecasted values, where 4 is for 2,3,8,12 weeks ahead forecast
            mvY    : 4*p matrix of actual values , where p is for 42 (minus subset if subset!=0) number of observations being considered in analysis 
        """
        for i in range(len(self.list_vYL)):
            if sForecMeth=='pick-up':
                #we dont update self.data
                f=i*4+2 #number of oos to predict
                iC=self.list_vYE[i].iloc[-1] # Cumulative bookings at hand
                vChat=self.data.iloc[-(f+1):, -11-i] # last year this time cumulative bookings window
                fM=vChat.iloc[-1]/vChat.iloc[0]
                self.list_vYhat[i]=pd.Series(fM*iC , index=[self.data.index[-1]])
            elif sForecMeth=='Prophet':
                    df=pd.DataFrame({'ds': self.data.columns, 'y': self.data.iloc[-1].values}) #prophet dataframe with target as y and time as ds
                    model=Prophet(
                                scaling='absmax',   
                                changepoint_prior_scale=0.05 , holidays_prior_scale=0.9,
                                yearly_seasonality=20
                            ) 
                    model.fit(df)
                    future = model.make_future_dataframe(periods=4, freq='MS')   #make the future df that will be used to predict _periods_ number of oos 
                    dfModel=model.predict(future)
                    self.vErrorIS=dfModel.yhat[:-4] 
                    for j in range (4):   #populate the list of predictions (list of vYhat) with a single ( end of the curve) prediction we predict only the end here, and not the full curve as we are not working with curves but discrete observations ( in contrast to FPCR )
                        self.list_vYhat[j]=pd.Series(dfModel.yhat.iloc[-4+j] , index=[self.data.index[-1]]) #forec = last 4 predictions 
                        # we dont add anything to data here , data stays static in contrast to FPCR update 
            
            
            else:
                # create a matrix that stores beta predictions
                # self.mBhat=np.zeros((4,iPC))
                # create a function data object of fully observed curves and partially observed target vector
                fdata=FData(data=self.data, vY=self.list_vY[i])
                forecast_date=self.list_vY[i].name  #the month we are forecasting, aka months of target vY
                #fdata.smooth_data(n_basis=n_basis)  #smooth the data
                
                fdata.demean_data()    # take the mean out of both data and target
                fdata.perform_FPCA(iPC) # perform FPCA on fully observed curves and partially observed portion of fully observed and unobserved target curve
                if sForecMeth=='FPCR_Prophet_update':
                    # perform FPCR with update - > the update is the transitory betas  # of FPCA on partially observed section of data + partially observed vY fed into Prophet as an regressor
                    vBhat, mBhatIS = self.forecast_betas_prophet(dfmBeta=fdata.dfmBeta, dfmBetaE=fdata.dfmBetaE)
                elif sForecMeth=='FPCR_Prophet':
                    # performs FPCR w/o update
                    vBhat, mBhatIS = self.forecast_betas_prophet(dfmBeta=fdata.dfmBeta, dfmBetaE=None)
               

                #self.mBhat[i]=vBhat # populate matrix of forecast betas for plotting purposes
                f=fdata.f  # number of nan observations in current vY, to predict = 2+4*i
                #smoothed and demeaned yHat , selects last -f elements of mPhi , only the last unoberved section
                vYhat=np.dot(fdata.mPhi[-f:],vBhat)  
                self.list_vY[i][-f:]=vYhat+fdata.sMyu[-f:] # remean
                self.list_vYhat[i]=self.list_vY[i][-f:]  # save only the forecasted vector
                self.data[forecast_date]=self.list_vY[i]  # add to data ( non smooth un-meaned)
                self.list_fdataY[i]=fdata
                
                # during first forecast, create vErrorIS = in sample forecast errors
                if i==0:
                    vErrorIS=np.dot(fdata.mPhi,mBhatIS.T) + fdata.sMyu.values.reshape(-1,1) - self.data.iloc[:,:-1] # -1 was just added OOS obs
                    self.vErrorIS=vErrorIS.iloc[-1,:] # we only need the end of the curve 
            

    def plot_data(df):
        data = []
        for column in df.columns:
            trace = go.Scatter(x=df.index, y=df[column], mode='lines', name=str(column))  # Convert column name to string
            data.append(trace)

        # Create layout
        layout = go.Layout(title='Data, days before departure', hovermode='closest')

        # Create figure
        fig = go.Figure(data=data, layout=layout)

        # Show plot
        fig.show()        
                    
    def plot_forecast_results(self, last_N=20):    
        """
        Plots Fitted, Predicted and Actual curves
        vYhat: vector, predicted part of the targed curve 
        vYe:   vector, observed part of the targed curve (in sample test set)
        vYl:   vector, unobserved part of the target curve (in sample training set)
        last_N: interger, number of observations from the end of the curve to display
        """ 
        
        # Plot actual values
        for i in range (len(self.list_vYhat)):
            x_values=self.data.index
            #plot actual values
            plt.plot(x_values[-last_N:], np.concatenate((self.list_vYE[i],self.list_vYL[i]))[-last_N:], label='Actual', color='blue',)
            
            # Plot predicted values
            # plt.plot(x_values[-last_N:], self.list_vYhat[i][-last_N:],'o',color='red' , label='Fitted and Predicted')
            plt.plot(self.list_vYhat[i],'o',color='red' , label='Predicted')

            plt.axvspan(x_values[-len(self.list_vYL[i])], x_values[-1], facecolor='gray', alpha=0.3)

            plt.xlabel('Before departure day')
            plt.ylabel('KPI value')
            
            L2_str = self.L2 if self.L2 is not None else ""
            L3_str = self.L3 if self.L3 is not None else ""
            L4_str = self.L4 if self.L4 is not None else ""
            plt.title(f" {i*4+2} weeks ahead {self.KPI} forecasts for {L2_str} {L3_str} {L4_str} flights departing in month of {self.list_vY[i].name.date().strftime('%Y-%m-%d')}")
            plt.legend({'Actual': 'blue', 'Fitted and Predicted': 'red', 'Forecasted': 'gray'}, loc='upper left')
            plt.show()
            
            if np.isnan(self.list_vYhat[i].iloc[-1]):
                print('ERROR: No predictions are available')
            else:
                print('End of the curve forecast error = '+ str((self.list_vYL[i].iloc[-1]-self.list_vYhat[i].iloc[-1])/self.list_vYL[i].iloc[-1]))
       
    def plot_reconciled_forecast_results(self, last_N=20):    
        """
        Plots Fitted, Predicted and Actual curves
        vYhat: vector, predicted part of the targed curve 
        vYe:   vector, observed part of the targed curve (in sample test set)
        vYl:   vector, unobserved part of the target curve (in sample training set)
        last_N: interger, number of observations from the end of the curve to display
        """ 
        
        self.list_vYL_rec=[self.vY0L_rec,self.vY1L_rec,self.vY2L_rec,self.vY3L_rec]
        # Plot actual values
        for i in range (len(self.list_vYhat)):
            x_values = self.data.index
            f=len(set(self.list_vYL_rec[i]))  #set so that we count only unique values : non functional methods produce horizontal lines instead of curves
            plt.plot(x_values[-last_N:], np.concatenate((self.list_vYE[i],self.list_vYL[i]))[-last_N:], label='Actual', color='blue',)
 
            #Plot reconciled values
            plt.plot(x_values[-last_N:][-f:], self.list_vYL_rec[i][-f:],'o',color='green' , label='Reconciled')
            
            plt.axvspan(x_values[-len(self.list_vYL[i])], x_values[-1], facecolor='gray', alpha=0.3)
            plt.xlabel('Before departure day')
            plt.ylabel('KPI value')
            
            L2_str = self.L2 if self.L2 is not None else ""
            L3_str = self.L3 if self.L3 is not None else ""
            L4_str = self.L4 if self.L4 is not None else ""
            plt.title(f" {i*4+2} weeks ahead {self.KPI} reconciled forecasts for {L2_str} {L3_str} {L4_str} flights departing in month of {self.list_vY[i].name.date().strftime('%Y-%m-%d')}")
            plt.legend({'Actual': 'blue', 'Fitted and Predicted': 'green', 'Reconciled':'red' , 'Out of Sample': 'gray' }, loc='upper left')
            plt.show()
            
            if np.isnan(self.list_vYL[i].iloc[-1]):
                print()
            else:
                print('End of the curve reconciled forecast error = '+ str((self.list_vYL_rec[i][-1]-self.list_vY[i].iloc[-1])/self.list_vY[i].iloc[-1]))
               
                    
    def plot_beta_predictions(self,iPC=3,vY_=0): 
        """ Plotting historical betas, predicted and true ( only in training use)

        Args:
            mBhat (_type_): _description_
            iPC (_type_): _description_
            vY_ (int): curve for which to plot the betas. if 0 then vY0, if 1 then vY1 etc.
        """
        _df_=self.raw_data.copy()  # includes also unobserved curve, thus to be used only in training
        vM = _df_.median(axis=1)
        _df_=_df_.sub(vM,axis=0)
        index_lookup=_df_.columns.get_loc(self.date)
        
        grid_points= [int(x) for x in _df_.index]
        data_matrix= _df_.T.values.tolist()
        
        fd = skfda.FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
        basis = skfda.representation.basis.MonomialBasis(domain_range=fd.domain_range, n_basis=10)
        smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
        fd=smoother.fit_transform(fd)
        
        #extract principal components and scores from df
        fpca = FPCA(n_components=iPC, centering=False)
        _mBeta_ = fpca.fit_transform(fd)            # pc scores
        
        for i in range(iPC):
                # Plot actual values
                plt.plot(self.raw_data.columns, _mBeta_[:,i], 'x',color='blue', linestyle='-',label='Actual PCS'+str(i+1))
                # Plot predicted values
                plt.plot(self.raw_data.columns[index_lookup], self.mBhat[vY_,i],'o',color='red', label='OLS predicted PCS'+str(i+1))
                plt.xlabel('Date')
                plt.ylabel('PC score')
                plt.title('Forecasted and Actual Betas for vY'+str(vY_))
                plt.legend()
                plt.show()


        