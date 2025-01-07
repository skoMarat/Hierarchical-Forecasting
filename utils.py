import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import os
from scipy.stats import ttest_1samp
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from scipy.stats import levene
from scipy.stats import kstest, norm
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from scipy.stats import f
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox


def ljung_box_test(vData, lags=7, fAlpha=0.05):
    """
    Perform the Ljung-Box Q test for autocorrelation.
    H0: Residuals are independently distributed (no autocorrelation)
    
    Args:
        vData (np.ndarray): The residuals or time series data to test.
        lags (int): The number of lags to include in the test. Default is 10.
        fAlpha (float): The significance level for the test. Default is 0.05.
        
    Returns:
        tuple: (Ljung-Box p-value, decision)
            - p-value (float): The p-value of the test.
            - decision (bool): True if no autocorrelation (fail to reject H0), 
              False if autocorrelation present (reject H0).
    """
    # Perform the Ljung-Box test
    p_value = acorr_ljungbox(vData, lags=[lags], return_df=False)['lb_pvalue'].item()
    
    # Determine if autocorrelation is present
    if p_value < fAlpha:
        return p_value, False  # Reject H0: Autocorrelation present
    else:
        return p_value, True  # Fail to reject H0: No autocorrelation

def jarque_bera_test(vData: np.ndarray, fAlpha=0.1):
    """
    Tests if a dataset follows a normal distribution using the Jarque-Bera test.
    H0: The data is normally distributed.

    Args:
        vData (np.ndarray): The data vector to test.
        fAlpha (float): The significance level for the test. Default is 0.1.

    Returns:
        tuple: A tuple containing:
            - p_value (float): The p-value from the Jarque-Bera test.
            - is_normal (bool): True if the data is normally distributed at the given significance level.
    """
    # Perform the Jarque-Bera test
    stat, p_value = jarque_bera(vData)
    
    # Determine normality
    if p_value < fAlpha: 
        return p_value, False  # Reject null hypothesis: data is not normal
    else:
        return p_value, True  # Fail to reject null hypothesis: data is normal

def durbin_watson_test(vData, model=None):
    """
    Perform the Durbin-Watson test for autocorrelation in residuals.
    H0: No first-order autocorrelation
    
    Args:
        vData (np.ndarray): The dependent variable (target data).
        model (sm.OLS object, optional): Fitted regression model. If not provided, will use `vData` to fit.
        
    Returns:
        tuple: (Durbin-Watson statistic, p-value, decision)
            - Durbin-Watson statistic: The test statistic for the DW test.
            - p-value: The p-value of the test.
            - decision: Boolean indicating if there is significant autocorrelation.
    """
    if model is None:
        # Fit an OLS model if not provided
        X = sm.add_constant(np.random.randn(len(vData)))  # Using random data for example, replace with actual predictors
        model = sm.OLS(vData, X).fit()

    # Perform the Durbin-Watson test
    dw_stat = durbin_watson(model.resid)

    # Durbin-Watson statistic ranges from 0 to 4:
    #  - Value around 2 indicates no first-order autocorrelation.
    #  - Value < 2 suggests positive autocorrelation.
    #  - Value > 2 suggests negative autocorrelation.
    
    # The null hypothesis is that there is no autocorrelation (DW ≈ 2).
    if dw_stat < 1.5 or dw_stat > 2.5:
        return dw_stat, False  # Reject null hypothesis (autocorrelation present)
    else:
        return dw_stat, True  # Fail to reject null hypothesis (no autocorrelation)

def f_test(vData: np.ndarray, fAlpha=0.1) -> bool:
    """
    Test if the data has constant variance using the F-test (i.e., equal variances in two groups).

    Args:
        vData (np.ndarray): The input data to test.
        fAlpha (float): Significance level for the test.

    Returns:
        tuple: A tuple containing the p-value and a boolean indicating 
               True if the variances are equal (constant variance), otherwise False.
    """
    # Split the data into two halves
    group1 = vData[:len(vData)//2]  # First half of the data
    group2 = vData[len(vData)//2:]  # Second half of the data

    # Compute variances
    var1 = np.var(group1, ddof=1)  # Variance of the first group
    var2 = np.var(group2, ddof=1)  # Variance of the second group

    # Compute the F-statistic
    f_stat = var1 / var2 if var1 > var2 else var2 / var1
    df1 = len(group1) - 1  # Degrees of freedom for group1
    df2 = len(group2) - 1  # Degrees of freedom for group2

    # Compute the p-value
    p_value = 2 * min(f.cdf(f_stat, df1, df2), 1 - f.cdf(f_stat, df1, df2))

    # Check if the variances are equal at the given significance level
    if p_value < fAlpha:
        # The variances are significantly different (not constant)
        return p_value, False
    else:
        # The variances are equal (constant)
        return p_value, True

def breusch_godfrey_test(vData, max_lags=7, model=None , fAlpha=0.1):
    """
    Perform the Breusch-Godfrey (BG) test for autocorrelation in residuals.
    H0: No serial correlation of any order upmto max_lags
    
    Args:
        vData (np.ndarray): The dependent variable (target data).
        max_lags (int): The number of lags to consider for autocorrelation testing.
        model (sm.OLS object, optional): Fitted regression model. If not provided, will use `vData` to fit.
        
    Returns:
        tuple: (Lagrange Multiplier statistic, p-value, decision)
            - Lagrange Multiplier statistic: The test statistic for the BG test.
            - p-value: The p-value of the test.
            - decision: Boolean indicating if there is significant autocorrelation.
    """
    if model is None:
        # Fit an OLS model if not provided
        X = sm.add_constant(np.random.randn(len(vData)))  # Using random data for example, replace with actual predictors
        model = sm.OLS(vData, X).fit()

    # Perform the Breusch-Godfrey test
    test_stat, p_value, _, _ = acorr_breusch_godfrey(model, nlags=max_lags)

    # Check if p-value is below the significance level (alpha=0.05)
    if p_value < fAlpha:
        return p_value , False  # Reject null hypothesis (autocorrelation present)
    else:
        return p_value , True  # Fail to reject null hypothesis (no autocorrelation)

def one_sample_t_test(vData: np.ndarray ,fAlpha=0.1) -> bool:
    """Test if the mean of the data is significantly different from zero.
    H0: Mean is not different from zero
    Args:
        vData (np.ndarray): The input data to test.

    Returns:
        bool: True if the mean of vData is not significantly different from zero, otherwise False.
    """
    # Perform one-sample t-test against the population mean of 0
    t_stat, p_value = ttest_1samp(vData, popmean=0)
    
    # If the p-value is less than 0.05, reject the null hypothesis (mean is significantly different from zero)
    if p_value < fAlpha:
        return p_value , False  # The mean is significantly different from zero
    else:
        return p_value, True   # The mean is not significantly different from zero
    
def levene_test(vData: np.ndarray,fAlpha=0.1) -> bool:
    """
    Test if the data has constant variance (i.e., equal variances in two groups).
    H0: all input samples are from populations with equal variances

    Args:
        vData (np.ndarray): The input data to test.

    Returns:
        bool: True if the variances are equal (constant variance), otherwise False.
    """
    # Split the data into two halves
    group1 = vData[:len(vData)//2]  # First half of the data
    group2 = vData[len(vData)//2:]  # Second half of the data

    # Perform Levene's test for equal variances
    stat, p_value = levene(group1, group2)

    # Check if the variances are equal at a 0.05 significance level
    if p_value < fAlpha:
        #The variances are significantly different (i.e., not constant)
        return p_value, False  # Variances are not equal
    else:
        #The variances are equal (i.e., constant).
        return p_value, True  # Variances are equal    

def displayMatrix(matrix:np.ndarray):
    plt.imshow(matrix,cmap='binary', interpolation='nearest')
    plt.show()  

def matrix_to_df(matrix: np.ndarray, row_names:list, column_names_1:list, column_names_2=None):  
    if column_names_2 is not None:
        tuples=list(itertools.product(column_names_1,column_names_2))
        multi_index = pd.MultiIndex.from_tuples(tuples, names=['Method', 'Relative'])
        df = pd.DataFrame(matrix, index=row_names, columns=multi_index)
    else:
        df = pd.DataFrame(matrix, index=row_names, columns=column_names_1)
        
    return df

def return_loss(mTrue, mPred , mW:np.ndarray, metric:str , slices:list ):
    """
    mY for scaled losses
    mW for any weighted losses
    
    """ 
    vSlices=np.zeros(len(slices))

    start=mTrue.shape[0]
    #per level
    for i,slice in enumerate(slices):
        end = start 
        start = end - slice
        
        mP=mPred[start:end,:]
        mT=mTrue[start:end,:]
        
        
        if metric=='MSE':
            loss=np.median(np.sum((mP-mT)**2,axis=0))
        elif metric=='RMSE':
            loss=np.sqrt(np.median(np.sum((mP-mT)**2,axis=0)))
        elif metric=='TSE':
            loss=np.sum((mP-mT)**2,axis=0)
        elif metric=='MAPE':
            loss=np.median(np.sum(np.abs(mP-mT)/(mT+1),axis=0))
        elif metric=='MPE':
            loss=np.median(np.sum((mP-mT)/(mT+1),axis=0))
        elif metric=='MWvarSE':
            vWeights=np.diag(mW)
            vW=vWeights[start:end]
            loss=np.median(np.sum(((mP-mT)**2)*vW[:, np.newaxis],axis=0))
        
        vSlices[i]=loss
        
    vSlices=vSlices[::-1]
    
    #the average
    if metric=='MSE':
        loss=np.median(np.sum((mPred-mTrue)**2,axis=0))
    elif metric=='RMSE':
        loss=np.sqrt(np.median(np.sum((mPred-mTrue)**2,axis=0)))
    elif metric=='MAPE':
        loss=np.median(np.sum(np.abs(mPred-mTrue)/(mTrue+1),axis=0))
    elif metric=='MPE':
        loss=np.median(np.sum((mPred-mTrue)/(mTrue+1),axis=0))
    elif metric=='TSE':
        loss=np.sum((mPred-mTrue)**2,axis=0)
    # elif metric=='RMSSE':
    #     loss=np.sqrt(np.mean((mTrue-mPred)**2 , axis=1)  / np.mean( (mY[:,1:] - mY[:,:-1])**2 , axis=1))    
    # elif metric=='WRMSSE':
    #     vRMSSE=np.sqrt(np.mean((mTrue-mPred)**2 , axis=1)  / np.mean( (mY[:,1:] - mY[:,:-1])**2 , axis=1))
    #     loss=np.diag(mW)*vRMSSE
    elif metric=='MWvarSE':  #mean weighted (variance) squared error
        vWeights=np.diag(mW)
        loss=np.median(np.sum(((mPred-mTrue)**2)*vWeights[:, np.newaxis],axis=0))    

    return np.hstack([vSlices,loss])

def diag_mat(rem=[], result=np.empty((0, 0))):
    """
    Recursively constructs a block-diagonal matrix by concatenating matrices from a list.

    This function takes a list of matrices and iteratively places them along the diagonal of a larger matrix.
    Each input matrix is added to the diagonal of the resulting matrix, and the off-diagonal elements
    are filled with zeros.

    Parameters:
    -----------
    rem : list of numpy arrays
        A list of 2D numpy arrays (matrices) to be placed on the diagonal of the resulting matrix.
        The matrices are added in the order they appear in the list.
    
    result : numpy array, optional, default=np.empty((0, 0))
        The current state of the result matrix (initially an empty matrix).
        The matrix is expanded recursively by adding the matrices from `rem` along the diagonal.

    Returns:
    --------
    numpy array
        The resulting block-diagonal matrix with matrices from `rem` placed along the diagonal.

    Example:
    --------
    rem = [np.array([[1]]), np.array([[2, 3], [4, 5]])]
    result = diag_mat(rem)
    print(result)
    # Output:
    # [[1. 0. 0. 0.]
    #  [0. 2. 3. 0.]
    #  [0. 4. 5. 0.]
    #  [0. 0. 0. 0.]]
    """
    
    if not rem:  # Base case: If no matrices are left, return the accumulated result.
        return result

    m = rem.pop(0)  # Pop the first matrix from the list.
    
    # Construct a new matrix with the current `result` and `m` on the diagonal.
    result = np.block([
        [result, np.zeros((result.shape[0], m.shape[1]))],  # Add zeros to the right of `result`.
        [np.zeros((m.shape[0], result.shape[1])), m]  # Add zeros below `result` and place `m` in the bottom-right.
    ])
    
    # Recursively call the function with the remaining matrices.
    return diag_mat(rem, result)

def sort_key(item):
    none_count = item.count(None)
    return (-none_count, item)  

def split_matrix(matrix, dLevels: dict):
    """
    Splits a matrix into smaller matrices based on row sizes derived from dLevels values.

    Args:
        matrix (np.ndarray): The input matrix to split.
        dLevels (dict): Dictionary with keys representing levels and values as the row sizes.

    Returns:
        list: A list of sub-matrices split based on dLevels values.
    """
    splits = []
    start = 0
    row_sizes = list(dLevels.values())  # Extract row sizes from dictionary
    for size in row_sizes:
        splits.append(matrix[start:start + size, :])  # Slice along rows
        start += size
    return splits

def time_converter(value, from_unit, to_unit):
    """
    Convert between time units: T, H, D, W, 
    and M, Q, SA, A.
    
    Units:
    - Higher frequency: 'T', 'H', 'D', 'W'
    - Lower frequency: 'M', 'Q', 'SA', 'A'
    """
    conversion_factors = {
        # Higher frequencies
        ('T', 'H'): 1 / 60,
        ('H', 'T'): 60,
        ('H', 'D'): 1 / 24,
        ('D', 'H'): 24,
        ('D', 'W'): 1 / 7,
        ('W', 'D'): 7,
        
        # Lower frequencies
        ('M', 'Q'): 1 / 3,
        ('Q', 'M'): 3,
        ('M', 'SA'): 1 / 6,
        ('SA', 'M'): 6,
        ('M', 'A'): 1 / 12,
        ('A', 'M'): 12,
        ('Q', 'SA'): 1 / 2,
        ('SA', 'Q'): 2,
        ('Q', 'A'): 1 / 4,
        ('A', 'Q'): 4,
        ('SA', 'A'): 1 / 2,
        ('A', 'SA'): 2,
    }

    # Check if the conversion is valid
    if (from_unit, to_unit) in conversion_factors:
        factor = conversion_factors[(from_unit, to_unit)]
        return int(value * factor)
    else:
        if from_unit==to_unit:
            return int(value)
        else:
            return f"Conversion from {from_unit} to {to_unit} is not supported."   

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

def create_cascading_matrix(list_of_lengths):
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

def getCVResults( h:int, iOoS:int, ddOutputs ,metric:str , slices: list , iters: int , rolling=True , relative=True):
    """ 
    
    Returns a dataframe of average relative metric , where average is computed over slices of the error vector
    as well as whole metric  (the bottom most row). 
    When rolling is True, then average is computed for all iterations and for all horizon windows
    When rolling is False, then average is computer only for the horizon h but still for every iteration
    When iters is None, average is computed for every iteration
    When iter is an integer then iters number of CV folds will be included in the average
    
    
    h (int) : horizon to fetch stats for = max( 1 , iOoS)
    iOoS (int) : maximum horizon that was used in CV , the horizon that was given to cv function #TODO changed to ddOutputs
    ddOutputs (dict) : of dictionaries, with reconciliation methods as keys, 
                       dOutputs[method] is a dictionary with keys as mY, mYhat and mYrec
    metric (str) : MAE or MAPE
    slices (list) : that designates indexes where different hierarchy borders begin 
    rolling (bool) : if True, calculates metris on a rolling basis h=1, h= 1-4 , h= 1-14 ...
    iters (int)  Number of iterations of CV to consider when getting the results
    relative (bool) : If True , relative to reference and base will be returned if False, absolute metric values will be presented
    """

    try:
        mW=ddOutputs['wls']['mW']
    except:
        mW=None
     
     
    #TODO   rows  must be dynamically populated based on data 
    lMethods=list(ddOutputs.keys())
    if 'wls_hvar' in ddOutputs.keys() : #then it is temporal
        rows=['W','D',"Average"]  #TODO
        reference='top_down_hp'
        lMethods.remove(reference)
    else: # its spatial
        rows= ['Total','State','Store','Cat.','Dept.','Average']  #TODO     
        reference='bottom_up'
        lMethods.remove(reference)
     
     
    if iters==None:
        mYtrue=ddOutputs[reference]['mYtrue']
        mYhat=ddOutputs[reference]['mYhat']
        mYtilde=ddOutputs[reference]['mYtilde'] 
    else:
        mYtrue=ddOutputs[reference]['mYtrue'][:,0+iOoS*(iters-1):iOoS+iOoS*(iters-1)] 
        mYhat=ddOutputs[reference]['mYhat'][:,0+iOoS*(iters-1):iOoS+iOoS*(iters-1)] # we assume iter is in range (1, M)
        mYtilde=ddOutputs[reference]['mYtilde'][:,0+iOoS*(iters-1):iOoS+iOoS*(iters-1)]
    if rolling==False:
        mYtrue=mYtrue[:,(h-1):][:,::iOoS]
        mYhat=mYhat[:,(h-1):][:,::iOoS]
        mYtilde=mYtilde[:,(h-1):][:,::iOoS]
    else:
        selected_indices = np.concatenate([np.arange(i, i+h) for i in range(0,mYhat.shape[1] , iOoS)])
        mYtrue=mYtrue[:,selected_indices]
        mYhat=mYhat[:,selected_indices]
        mYtilde=mYtilde[:,selected_indices]
    
        
    #SELECT METRIC
    vLossHat=return_loss(mYtrue,mYhat, mW=mW, metric=metric, slices=slices)
    vLossTilde_bu=return_loss(mYtrue,mYtilde, mW=mW, metric=metric, slices=slices)     
    
    if relative==True:
        mResults=np.zeros((vLossTilde_bu.shape[0], 2*(len(lMethods)+1)))
        mResults[:,0]=(1-vLossTilde_bu/vLossHat)*100
        mResults[:,1]=(1-vLossTilde_bu/vLossTilde_bu)*100
    elif relative==False:
        mResults=np.zeros((len(slices)+1,len(lMethods)+1))  
        mResults[:,0]=vLossTilde_bu

    #DO THE SAME FOR THE REST OF THE METHODS
    for j,sWeightType in enumerate(lMethods):
        if iters==None:
            mYtilde=ddOutputs[sWeightType]['mYtilde'] 
        else:
            mYtilde=ddOutputs[sWeightType]['mYtilde'][:,0+iOoS*(iters-1):iOoS+iOoS*(iters-1)]

        if rolling==False:
            mYtilde=mYtilde[:,(h-1):][:,::iOoS]
        else:
            mYtilde=mYtilde[:,selected_indices]

        vLossTilde=return_loss(mYtrue,mYtilde,mW=mW, metric=metric, slices=slices)     
        if relative==True:
            mResults[:,0+2*(j+1)]=(1-vLossTilde/vLossHat)*100
            mResults[:,1+2*(j+1)]=(1-vLossTilde/vLossTilde_bu)*100
        elif relative==False:
            mResults[:,j+1]=vLossTilde

    if relative==True:
        dfResults = matrix_to_df(np.round(mResults,2),
                        rows, 
                        [reference]+lMethods,
                        ["Base",reference])
    else:
        dfResults = matrix_to_df(np.round(mResults,2),
                        rows, 
                        [reference]+lMethods)
        
    return dfResults
    
