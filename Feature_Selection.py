# -*- coding: utf-8 -*-
"""
This code contains all the feature selection function, using Random Forest, LassoC, ElastikNet and recursive feature elimination


@author: laramos
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
"""
Feature Selection
"""
def SelectFeatures_Recursive(X,y,cv):
    """
    This functions returns only the features selected by the method using the threshold selected.
    We advise to run this function with several thresholds and look for the best, put this function inside a loop and see how it goes
    Suggestions for the range of t, thresholds = np.linspace(0.00001, 0.1, num=10)
    Input: 
        X=training set
        y=training labels
        T=threshold selected
        which method= 'rfc', 'lasso', 'elastik'
        cv= number of cross validation iterations
    Output:
    Boolean array with the selected features,with this you can X=X[feats] to select only the relevant features
    """

                
    lr=LogisticRegression()
        
    rfe = RFECV(lr,cv=cv,n_jobs=-1)
    rfe = rfe.fit(X, y)
    feats=rfe.support_
        
         
    return(feats)

def SelectFeatures(X,y,T,which,cv):
    """
    This functions returns only the features selected by the method using the threshold selected.
    We advise to run this function with several thresholds and look for the best, put this function inside a loop and see how it goes
    Suggestions for the range of t, thresholds = np.linspace(0.00001, 0.1, num=10)
    Input: 
        X=training set
        y=training labels
        T=threshold selected
        which method= 'rfc', 'lasso', 'elastik'
        cv= number of cross validation iterations
    Output:
    Boolean array with the selected features,with this you can X=X[feats] to select only the relevant features
    """
    alphagrid = np.linspace(0.001, 0.99, num=cv)
    if (which=='rfc'):
        rf=RandomForestClassifier(class_weight='balanced') 
    else:
        if (which=='lasso'):
            rf=LassoCV(alphas=alphagrid, cv = cv)
        else:
            if (which=='elastik'):
                rf=ElasticNetCV(alphas=alphagrid, cv = cv)
                
    rf.fit(X,y)
    sfm = SelectFromModel(rf, threshold=T)
    sfm.fit(X,y)                                            
    feats=sfm.get_support()
    return(feats)




def SelectFeaturesTestTresh(X,y,T,which,cv):
    """
    This function is used to check what are the important features, it records the features
    with the variable histoffeats over N splits, creating a histogram
    if you split in training and testing and call this function you can do the same in the main function and get a histogram with more iterations
    Input:
        X=training set
        y=training labels
        T=threshold selected
        which method= 'rfc', 'lasso', 'elastik'
        cv= number of cross validation iterations
    Output: 
       histoffeats: a histogram where each postion is a feature and each number is the number of times selected
       meanauc: the mean auc over all cv iterations
       stdauc: standard deviation of the mean auc
       importance: weights given by the models 
       
    """
    alphagrid = np.linspace(0.001, 0.99, num=cv)
    
    if (which=='rfc'):
        rf=RandomForestClassifier() 
    else:
        if (which=='lasso'):
            rf=LassoCV(alphas=alphagrid, cv = cv)
        else:
            if (which=='elastik'):
                rf=ElasticNetCV(alphas=alphagrid, cv = cv)
    #fit the method before using in the feature selection fucntion            
    rf.fit(X,y)
       
    if (which=='rfc'):
        importance=rf.feature_importances_
    else:
        importance=abs(rf.coef_)
    
    
    splits=cv
    ss = ShuffleSplit(n_splits=splits, test_size=0.10,random_state=1)

    roc_auc=np.zeros(splits)    
    i=0
    numfeats=np.zeros(splits)

    histoffeats=np.zeros(X.shape[1],dtype='int32')

    for train, test in ss.split(X):
        X_train=X[train]
        y_train=y[train]
        X_test=X[test]
        y_test=y[test]
   
        lr=LogisticRegression(class_weight='balanced')
        sfm = SelectFromModel(rf, threshold=T)
        sfm.fit(X_train,y_train)                                            
        
        arr=sfm.get_support()#get features
       
        X_train=X_train[:,arr]
        X_test=X_test[:,arr]
        
        
        histoffeats=histoffeats+arr
       # print(X_train.shape[1])
        
        numfeats[i]=X_train.shape[1]
        if X_train.shape[1]>0:
            lr.fit(X_train,y_train)  
            predict = lr.fit(X_train,y_train).predict(X_test)
            fpr, tpr, thresholds_ = roc_curve(y_test, predict[:])
            roc_auc[i] = auc(fpr, tpr) 
        i=i+1
    meanauc=np.mean(roc_auc)
    stdauc=np.std(roc_auc)

    return(histoffeats,meanauc,stdauc,importance)

def Select_Recursive_Features(X,Y,cv):
  """
  This function returns an overview of the best number of features found using backward selection and LR
  This cannot be used to select the features, this is just for visualization
    Input:
        X=training set
        y=training labels
        cv= number of cross validation iterations
    Output: 
       histoffeats: a histogram where each postion is a feature and each number is the number of times selected
       meanauc: the mean auc over all cv iterations
       stdauc: standard deviation of the mean auc
       
    
  """
   
  ss = ShuffleSplit(n_splits=cv, test_size=0.10,random_state=1)

  histoffeats=np.zeros(X.shape[1],dtype='int32')
  i=0
  for train, test in ss.split(X):
              
        roc_auc=np.zeros(X.shape[1])


        X_train=X[train]
        Y_train=Y[train]
        X_test=X[test]
        Y_test=Y[test]

        lr=LogisticRegression()
        
        rfe = RFECV(lr,cv=2,n_jobs=-1)
        rfe = rfe.fit(X_train, Y_train)
        arr=rfe.support_
        X_train=X_train[:,arr]
        X_test=X_test[:,arr]
        histoffeats=histoffeats+arr

        clf=LogisticRegression(class_weight='balanced')
        clf.fit(X_train,Y_train)  
        predict = clf.predict_proba(X_test)[:, 1]
        
        fpr, tpr, thresholds_ = roc_curve(Y_test, predict)
        
        roc_auc[i] = auc(fpr, tpr)    
        i=i+1
  meanauc=np.mean(roc_auc)
  stdauc=np.std(roc_auc)      

           
  return(histoffeats,meanauc,stdauc)

def Change_One_Hot(frame,vals_mask):
    """
    This function one-hot-encode the features from the vals_mask and returns it as numpy array
    Input:
        frame: original frame with variables
        vals_mask: array of string with the names of the features to be one-hot-encoded [['age','sex']]
    Ouput:
        Result: One-hot-encoded feature set in pd.frame format
    """
    new_frame=frame[vals_mask]
    X_vars=np.array(new_frame,dtype='float64')
    rf_enc = OneHotEncoder()
    rf_enc.fit(X_vars)
    Result=rf_enc.transform(X_vars)
    Result=Result.toarray()
    Result=np.array(Result,dtype='float64')
   
    feat_ind=np.array((np.max(X_vars,axis=0)-np.min(X_vars,axis=0)+1),'int32')
    cols=list()    
    for i in range(0,feat_ind.shape[0]):
        for j in range(0,feat_ind[i]):
            cols.append(vals_mask[i]+str(j))
    
    Result=pd.DataFrame(Result,columns=cols)   
        
    return(Result)

def Encode_for_Logit(X_encode,Y,pos):
    """
    This function one-hot-encode the features from the feats selected for the logistic regression code, since it uses data without one-hot-encoding
    Input:
        X_encode: The normal dataset without any feature encoding but only the features selected
        Y: Label
        pos: positions of features to be fixed        
    Ouput:
        Result: One-hot-encoded feature set in numo=py format
    """
    to_delete=list()
    for j in range(0,pos.shape[0]):
        if pos[j,0]==1:
            rf_enc = OneHotEncoder()
            Enc=X_encode[:,pos[j,1]]
            Enc=Enc.reshape(-1,1)
            rf_enc.fit(Enc,Y)
            Result=rf_enc.transform(Enc).toarray()
            X_encode=np.concatenate((X_encode,Result),axis=1)
            to_delete.append(pos[j,1])
     
    X_encode=np.delete(X_encode,np.array(to_delete),axis=1)
    return(X_encode)

def Fix_for_logit(X,Y,cols,feats,vals_mask):
    """
    This function one-hot-encode the features from the feats selected for the logistic regression code, since it uses data without one-hot-encoding
    Input:
        X: The normal dataset without any feature encoding
        Y: Label
        cols: name of the columnsof the dataset
        feats: features selected by the method feature selection
        vals_mask: mask with the names of variables thathave to be one-hot-encoded
    Ouput:
        Result: One-hot-encoded feature set in numpy format
    """
    cols_feats=cols[feats]
    cols_feats2=np.array(cols_feats)
      
      #this variable tells if a feature that has to be fixed is present or not in the first postion and the second is where it is
    pos_mask=np.zeros((7,2),dtype='int32')      

    for i in range(0,len(vals_mask)):
          if np.array(vals_mask[i]) in cols_feats2:
              pos_mask[i,0]=1 
              pos_mask[i,1]=np.where(cols_feats2==vals_mask[i])[0]
              
    X_logit=Encode_for_Logit(X[:,feats],Y,pos_mask)
    return(X_logit)