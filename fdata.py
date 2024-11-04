import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import  skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler

class FData:
    """
    (Functional) data class, includes methods for handling data, putting data into FDA format , smoothing data, representing data in functional principal components (fpc)
    generating betas that represent data in iPC number of principal components. Forecasting betas,
    and forecasting the curve as well as visualizations
    """
    def __init__(self, data, vY ):
        self.data=data  # fully observed curve
        self.f=vY.isna().sum()  # number of missing observations in current vY, to predict
        self.vY=vY.dropna() # partially observed curve
        # self.vM=
        # self.vMe=
        # self.vMl=
        
        self.dfmBeta=None
        self.mPhi=None
        self.mErrorFPCA=None
        self.fd=None
        self.fpca=None
        
        self.dataE=pd.concat([self.data, self.vY],axis=1).dropna()
        self.dfmBetaE=None
        self.mPhiE=None
        self.mErrorFPCAE=None
        self.fdE=None
        self.fpcaE=None
        
        self.dataL=self.data[-self.f:]    
        self.dfmBetaL=None
        self.mPhiL=None
        self.mErrorFPCAL=None
        self.fdL=None
        self.fpcaL=None
        
        

    def smooth_data(self, n_basis):
        """
        smoothes the data ( fully observed curves), smoothes vY 
        
        inputs:
        n_basis: number of basis curves to use in smoothing
        """
        #Create data objects
        grid_points= [int(x) for x in self.data.index]
        data_matrix= self.data.T.values.tolist()
        fd = skfda.FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
        basis = skfda.representation.basis.MonomialBasis(domain_range=fd.domain_range, 
                                                                n_basis=n_basis)
        smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
        fd=smoother.fit_transform(fd)
        grid=fd.data_matrix
        self.data = pd.DataFrame(grid.reshape((self.data.shape[1], self.data.shape[0])).T, columns=self.data.columns , index=self.data.index)
        
        #Create dataE objects
        grid_points= [int(x) for x in self.vY.index]
        data_matrix=self.vY.values
        fd = skfda.FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
        basis = skfda.representation.basis.MonomialBasis(domain_range=fd.domain_range, n_basis=n_basis)
        smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
        fd=smoother.fit_transform(fd)
        grid=fd.data_matrix
        coords=fd.grid_points
        v = grid[0]
        v=[item[0] for item in v]
        self.vY=pd.Series(np.array(v),index=self.vY.index)
        self.dataE=pd.concat([self.data, self.vY],axis=1).dropna()
        
        # #create dataL objects
        # grid_points= [int(x) for x in self.dataL.index]
        # data_matrix=self.dataL.values
        # fd = skfda.FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
        # basis = skfda.representation.basis.MonomialBasis(domain_range=fd.domain_range, n_basis=n_basis)
        # smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
        # fd=smoother.fit_transform(fd)
        # grid=fd.data_matrix
        # coords=fd.grid_points
        # v = grid[0]
        # v=[item[0] for item in v]
        # self.vY=pd.Series(np.array(v),index=self.vY.index)
        # self.dataE=pd.concat([self.data, self.vY],axis=1).dropna()

    def return_remean_data(self):
        """
        Remeans the data . Executes only if data is demeaned_already
        """
        try:
            return self.data.add(self.vM , axis=0)   
        except:
            print('Data has not been demeaned ')
            
            
    def demean_data(self):
        """
        demeans the data ( observed curves)
        """    
        self.sMyu=self.data.mean(axis=1)
        self.data=self.data.sub(self.sMyu,axis=0)
        self.vY=self.vY - self.sMyu #[:-len(self.vY)] 
        
        self.sMyu_e=self.dataE.mean(axis=1)
        self.dataE=self.dataE.sub(self.sMyu_e,axis=0)
        self.vYe=self.vY - self.sMyu_e #[:-len(self.vY)]
        
        self.sMyu_l=self.dataL.mean(axis=1)
        self.dataL=self.dataL.sub(self.sMyu_l,axis=0)
        self.vYl=self.vY - self.sMyu_l #[:-len(self.vY)]
        

    def perform_FPCA(self,iPC):
        """
        performs Principal component Analysis and populates mBeta, mPhi, mErrorFPCA, fd, fpca objects 
        as well as mBetaE, mPhiE  objects which are FPCA objects from partially observed curve
        mBeta: matrix, empirically obtained principal components extracted from fully observed curves
        mPhi:  matrix, principal component vectors of fully observed curves
        mErrorFPCA: matrix, error of FPCA, unexplained part of Variance Covariance Matrix.
        fd: skfda object, data grid
        fpca: skfda object , fitted data grid
        
        """
        grid_points= [int(x) for x in self.data.index]
        data_matrix= self.data.T.values.tolist() 
        self.fd = skfda.FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
        #extract principal components and scores from data
        self.fpca = FPCA(n_components=iPC, centering=False)
        mBeta = self.fpca.fit_transform(self.fd)            # pc scores
        self.dfmBeta = pd.DataFrame(mBeta,
                               index=pd.date_range(start=self.data.columns[0], 
                               end=self.data.columns[-1], freq='MS'))
        self.mPhi = self.fpca.components_.data_matrix.T[0]  # pc vectors
        #extract FPCA error
        self.mErrorFPCA=self.fpca.components_.data_matrix.T[0]  # pc vectors
        
        
        # partially observed part FPCA for dataE
        grid_points= [int(x) for x in self.dataE.index]
        data_matrix= self.dataE.T.values.tolist() 
        self.fdE = skfda.FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
        self.fpcaE = FPCA(n_components=iPC, centering=False)
        mBetaE=self.fpcaE.fit_transform(self.fdE)
        self.dfmBetaE=pd.DataFrame(mBetaE,
                               index= pd.date_range(start=self.data.columns[0], 
                               end=self.dfmBeta.index[-1]+relativedelta(months=1), freq='MS'))
        self.mPhiE=self.fpca.components_.data_matrix.T[0]  # pc vectors
        self.mErrorFPCAE=self.fpca.components_.data_matrix.T[0]  # pc vectors