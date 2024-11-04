import pandas as pd
import numpy as np
import os
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import plotly.graph_objs as go
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
import logging
import cmdstanpy
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

class Forecast_PCR:
    def __init__(self, dfData: pd.DataFrame, iOoS:int ):
        """
        dfData          : dataframe with columns datetime indeces in ascending order
                        : rows are the knots of the curves from left to the right of the curve
        iOoS            : integer number of oos observations (curves) to forecast
        iPC             : integer number of principal components to use 
        """
        self.dfData=dfData    # data
        self.iOoS=iOoS  # number of OoS forecast to generate in the same granularity as srY
        self.srYhat=None     # forecasted series TODO
        self.sFreq=self.dfData.index.inferred_freq
        self.mY=dfData.values
        self.time_index=dfData.columns

    def forecast(self):
        mYdemeaned=self.mY-self.mY.mean(axis=1)[:, np.newaxis]
        pca=PCA(n_components=3)
        mPhi = pca.fit_transform(mYdemeaned)

        
        
        
    
    
        
            
    
    
