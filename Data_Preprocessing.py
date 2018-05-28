# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:57:40 2017

@author: laramos
"""
import numpy as np
import iari as imp
import pandas as pd
#from fancyimpute import MICE

def Fix_Dataset(frame,images):
    
    names=np.array(frame['StudySubjectID'],dtype='int32')
    frame=frame.drop('StudySubjectID',axis=1)
    
    rows=frame.shape[0]
    
    #checks where data is equal to NULL, so we can change for -1 
    
    cols=frame.columns
 
    for i in range(0,rows):
        frame.at[i,'SAH_vol_ml']=frame.at[i,'SAH_vol_ml'].replace(',','.')
    
    for i in range(0,rows):
        frame.at[i,'TIME_ICTUS_CTSCANDT']=frame.at[i,'TIME_ICTUS_CTSCANDT'].replace(',','.')
    
    for i in range(0,rows):
        for j in range(53,frame.shape[1]):
            if (pd.isnull(frame.at[i,cols[j]])==False):
                frame.at[i,cols[j]]=frame.at[i,cols[j]].replace(',','.')    
                
    #Here I replace all the weird missing values for a common nan value
    for i in range(0,rows):
        if (frame.at[i,'ADMISSION_GCS_V_E1_C1']=='T'):
            frame.at[i,'ADMISSION_GCS_V_E1_C1']=6
        if (frame.at[i,'ADMISSION_GCS_V_E1_C1']=='A'):
            frame.at[i,'ADMISSION_GCS_V_E1_C1']=7
    
    for i in range(0,rows):
        if (frame.at[i,'ADMISSION_GCS_V_AMC_E1_C1']=='T'):
            frame.at[i,'ADMISSION_GCS_V_AMC_E1_C1']=6
        if (frame.at[i,'ADMISSION_GCS_V_AMC_E1_C1']=='A'):
            frame.at[i,'ADMISSION_GCS_V_AMC_E1_C1']=7
    
    for i in range(0,rows):
        if (frame.at[i,'ADMISSION_GCS_TOTAL_E1_C1']==88):
            frame.at[i,'ADMISSION_GCS_TOTAL_E1_C1']=-1
        
    frame=frame.drop('TIME_TILL_DEATH',axis=1)
    frame=frame.drop('TREATMENT_DCI_E1_C1',axis=1)
    frame=frame.drop('Poor_outcome',axis=1)
    frame=frame.drop('mRS_Final',axis=1)
    frame=frame.drop('DISCHARGE_DISEASED_E1_C1',axis=1)
    frame=frame.drop('ANEURYSM_LOCATIONOTHER_E1_C1_1',axis=1)
    frame=frame.drop('WidthQuint',axis=1)
    frame=frame.drop('AgeQuint',axis=1)
    frame=frame.drop('SAHvolQuint',axis=1)
    
    dataread = np.array(frame,dtype='float64')
    cols=frame.columns
    frame=frame.drop('Clin_DCI',axis=1)

    
    cols_l=pd.Index.tolist(cols)
    ind1=cols_l.index('Clin_DCI')
    
    for i in range(0,dataread.shape[0]):
        for j in range(0,dataread.shape[1]):
            if (dataread[i,j]==-2146826288 or dataread[i,j]==999 or dataread[i,j]==99):
                print(dataread[i,j])
                dataread[i,j]=np.nan
                #dataread[i,j]=-1
            
    n=dataread.shape[0]-1
    cont=0
    i=0
    #here I delete all the scans that the images were not included, I have to add them and rerun everything =X
    while (i<n):
            if (np.isnan(dataread[i,dataread.shape[1]-1])):
                dataread=np.delete(dataread,i,0)
                names=np.delete(names,i,0)
                i=i-1
                n=n-1
                cont=cont+1
            i=i+1
    
    Y=dataread[:,ind1]
    dataread=np.delete(dataread,ind1,1)
    if (images==False):
        dataread=dataread[:,0:ind1]
        cols=cols[0:ind1]
    
    Y = np.array(Y)
    Y=Y.astype('float64')
    
    X=np.array(dataread)        
    X=X.astype('float64')
    #mice = MICE()
   # X = mice.complete(np.asarray(X, dtype=float))
    X=imp.IARI(X,Y)#False)
    
    return(X,Y,cols,names)
    
    

