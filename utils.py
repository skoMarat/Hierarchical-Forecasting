import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import os


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
    for i,slice in enumerate(slices):
        end = start 
        start = end - slice
        
        mP=mPred[start:end,:]
        mT=mTrue[start:end,:]
        
        if metric=='MSE':
            loss=np.mean(np.sum((mP-mT)**2,axis=0))
        elif metric=='RMSE':
            loss=np.sqrt(np.mean(np.sum((mP-mT)**2,axis=0)))
        elif metric=='TSE':
            loss=np.sum((mP-mT)**2,axis=0)
        elif metric=='MAPE':
            loss=np.mean(np.sum(np.abs(mP-mT)/mT,axis=0))
        elif metric=='MWvarSE':
            vWeights=np.diag(mW)
            vW=vWeights[start:end]
            loss=np.mean(np.sum(((mP-mT)**2)*vW[:, np.newaxis],axis=0))
        
        vSlices[i]=loss
        
    vSlices=vSlices[::-1]
    
    if metric=='MSE':
        loss=np.mean(np.sum((mPred-mTrue)**2,axis=0))
    elif metric=='RMSE':
        loss=np.sqrt(np.mean(np.sum((mPred-mTrue)**2,axis=0)))
    elif metric=='MAPE':
        loss=np.mean(np.sum(np.abs(mPred-mTrue)/mTrue,axis=0))
    elif metric=='TSE':
        loss=np.sum((mPred-mTrue)**2,axis=0)
    # elif metric=='RMSSE':
    #     loss=np.sqrt(np.mean((mTrue-mPred)**2 , axis=1)  / np.mean( (mY[:,1:] - mY[:,:-1])**2 , axis=1))    
    # elif metric=='WRMSSE':
    #     vRMSSE=np.sqrt(np.mean((mTrue-mPred)**2 , axis=1)  / np.mean( (mY[:,1:] - mY[:,:-1])**2 , axis=1))
    #     loss=np.diag(mW)*vRMSSE
    elif metric=='MWvarSE':  #mean weighted (variance) squared error
        vWeights=np.diag(mW)
        loss=np.mean(np.sum(((mPred-mTrue)**2)*vWeights[:, np.newaxis],axis=0))
        

    return np.hstack([vSlices,loss])



def return_vAverageRelativeMetric(vMetric:np.ndarray, vMetricRelative:np.ndarray, slices:list):  #TODO take out the vRelativeMetric into getCVResults
    vRealativeMetric=vMetric/vMetricRelative
    vAverageRelativeMeanMetric=[]

    start=len(vRealativeMetric)
    for slice in slices:
        end = start 
        start = end - slice
        vAverageRelativeMeanMetric.append(np.mean(vRealativeMetric[start:end]))
        
    vAverageRelativeMeanMetric=(np.ones(len(vAverageRelativeMeanMetric))-vAverageRelativeMeanMetric[::-1])*100
    ARmetric=(1-np.mean(vRealativeMetric))*100
    
    return np.hstack([vAverageRelativeMeanMetric,ARmetric])


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
        return value * factor
    else:
        if from_unit==to_unit:
            return value
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

def getCVResults( h:int, dOutputs ,metric:str , slices: list , iters: int , rolling=True , relative=True):
    """ 
    
    Returns a dataframe of average relative metric , where average is computed over slices of the error vector
    as well as whole metric  (the bottom most row). 
    When rolling is True, then average is computed for all iterations and for all horizon windows
    When rolling is False, then average is computer only for the horizon h but still for every iteration
    When iters is None, average is computed for every iteration
    When iter is an integer then iters number of CV folds will be included in the average
    
    
    h (int)
    dOutputs (dict)
    metric (str) : MAE or MAPE
    slices (list) : that designates indexes where different hierarchy borders begin 
    rolling (bool) : if True, calculates metris on a rolling basis h=1, h= 1-4 , h= 1-14 ...
    iters (int)  Number of iterations of CV to consider when getting the results
    
    """

    lMethods=list(dOutputs.keys())[1:]
    mW=dOutputs['wls']['mW']
     
    if iters==None:
        mYtrue=dOutputs["bottom_up"]['mYtrue']
        mYhat=dOutputs["bottom_up"]['mYhat']
        mYtilde=dOutputs["bottom_up"]['mYtilde'] 
    else:
        # if rolling==True:
        #     mYtrue=dOutputs["bottom_up"]['mYtrue'][:,:28*iters] #TODO  28 needs to be dynamic
        #     mYhat=dOutputs["bottom_up"]['mYhat'][:,:28*iters]
        #     mYtilde=dOutputs["bottom_up"]['mYtilde'][:,:28*iters]
        # elif rolling==False:
            mYtrue=dOutputs["bottom_up"]['mYtrue'][:,0+28*(iters-1):28+28*(iters-1)] #TODO  28 needs to be dynamic
            mYhat=dOutputs["bottom_up"]['mYhat'][:,0+28*(iters-1):28+28*(iters-1)] # we assume iter is in range (1, M)
            mYtilde=dOutputs["bottom_up"]['mYtilde'][:,0+28*(iters-1):28+28*(iters-1)]
    if rolling==False:
        mYtrue=mYtrue[:,(h-1):][:,::28]
        mYhat=mYhat[:,(h-1):][:,::28]
        mYtilde=mYtilde[:,(h-1):][:,::28]
    else:
        selected_indices = np.concatenate([np.arange(i, i+h) for i in range(0,mYhat.shape[1] , 28)])
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
            mYtilde=dOutputs[sWeightType]['mYtilde'] 
        else:
            # if rolling==True:
            #     mYtilde=dOutputs[sWeightType]['mYtilde'][:,:28*iters]
            # elif rolling==False:
                mYtilde=dOutputs[sWeightType]['mYtilde'][:,0+28*(iters-1):28+28*(iters-1)]

        if rolling==False:
            mYtilde=mYtilde[:,(h-1):][:,::28]
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
                        ['Total','State','Store','Cat.','Dept.','Average'], #TODO 
                        ["bottom_up"]+lMethods,
                        ["Base","BU"])
    else:
        dfResults = matrix_to_df(np.round(mResults,2),
                        ['Total','State','Store','Cat.','Dept.','Average'], #TODO 
                        ["bottom_up"]+lMethods)
        
    return dfResults
    
