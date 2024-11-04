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
    mW=dOutputs['diag']['mW']
     
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
                        ["BU"]+lMethods,
                        ["Base","BU"])
    else:
        dfResults = matrix_to_df(np.round(mResults,2),
                        ['Total','State','Store','Cat.','Dept.','Average'], #TODO 
                        ["BU"]+lMethods)
        
    return dfResults
    
