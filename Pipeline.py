# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:00:03 2017

@author: laramos
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

import time
from sklearn.linear_model import (LinearRegression, Ridge, 
                                 Lasso, RandomizedLasso)

from scipy import interp                      
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

import warnings
from sklearn.externals import joblib   
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
import os
import Data_Preprocessing as dp
import Feature_Selection as fs
#import statsmodels.imputation.mice as mice
#import statsmodels.api as sm
import Methods_Prospective as mt
#from fancyimpute import MICE
#import iari as imp
#import MICE_code as mice
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

import scipy.stats as stats
import pylab 


#def Create_statistics(X1,cols):
    

"""
-----------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    
    pathtrain ='E:\Prospective_dataset\ClinicalDataWithImage.csv'
    path_image_data='E:\\DCI Prediction\\Data\\Image_data'
    
    frame = pd.read_csv(pathtrain,sep=';')
    image_feats=True
    [X1,Y,cols,names]=dp.Fix_Dataset(frame,image_feats) #false for no images features
    Features=mt.Connect_Image_Features(names,path_image_data)
    cols=pd.Index.tolist(cols)
  
    #important columns according to feature selection with RFC
    ind1=cols.index('SAH_vol_ml')
    ind2=cols.index('DIAGNOSIS_FISHER4_E1_C1_IPH')
    ind3=cols.index('TIME_ICTUS_CTSCANDT')
    ind4=cols.index('Age')
    ind5=cols.index('ADMISSION_GCS_TOTAL_AMC_E1_C1')
    ind6=cols.index('ANEURYSM_LENGTH_E1_C1_1')
    ind7=cols.index('DIAGNOSIS_FISHER4_E1_C1_SDH')
    ind8=cols.index('ANEURYSM_WIDTH_E1_C1_1')
    ind9=cols.index('TREATMENT_E1_C1')
    ind10=cols.index('ANEURYSM_LOCATION_E1_C1_1')
    
    #ind11=cols.index('HISTORY_SMOKING_E1_C1')
    #ind12=cols.index('REBLEEDAANTAL')
    #ind13=cols.index('PREVIOUS_iMRS_E1_C1')
    #ind14=cols.index('ADMISSION_GCS_TOTAL_E1_C1')
    
    #X=X1[:,[ind1,ind2,ind3,ind4,ind5,ind6,ind7,ind8,ind9,ind10,ind11,ind12,ind13,ind14]]
    X=X1[:,[ind1,ind2,ind3,ind4,ind5,ind6,ind7,ind9,ind10]]
      
    
    """
    ------------------------------------------------------------------------------------------------------------------------------
    gridsplit is used for the grid search, , so every combination of parameters is tested at least 4 times on different partds po the dataset
    dont make gridsplit too big, there is already a lot of permutation in gridsearch
    Splits is the number of times the whole pipeline will run, 100 is ok, but it will take some time
    """

    GridSplit=10
    splits=100
    
    aucb=np.zeros(splits)
    auc2b=np.zeros(splits)
    l=0
    
    skf = StratifiedShuffleSplit(n_splits=splits, test_size=0.25,random_state=1)
    #skf = ShuffleSplit(n_splits=splits, test_size=0.25,random_state=1)
    
    RunSVM=False
    RunRFC=True
    RunLogit=True
    RunNN=False
 
    TotalFeatures=X.shape[1]

    Testsvm=np.zeros((splits,2))
    TestForest=np.zeros((splits,2))
    TestLogit=np.zeros((splits,2))
    Testnn=np.zeros((splits,2))
    
    Accsvm=np.zeros((splits))
    Accrfc=np.zeros((splits))
    Accnn=np.zeros((splits))
    AccLog=np.zeros((splits))
    
    senssvm=np.zeros((splits))
    sensrfc=np.zeros((splits))
    sensnn=np.zeros((splits))
    senslog=np.zeros((splits))
    
    specsvm=np.zeros((splits))
    specrfc=np.zeros((splits))
    specnn=np.zeros((splits))
    speclog=np.zeros((splits))
    

    gridsvm = []
    gridLogit = []
    gridrfc = []
    gridNN = []   
    
    size=X.shape

   
    itera=0
    mean_tprs = 0.0
    mean_fprs = np.linspace(0, 1, 100)
    mean_tprr = 0.0
    mean_fprr = np.linspace(0, 1, 100)
    mean_tprn = 0.0
    mean_fprn = np.linspace(0, 1, 100)
    mean_tprlog = 0.0
    mean_fprlog = np.linspace(0, 1, 100)
    
    thresholds = np.linspace(0.00001, 0.1, num=10)
    
    if not os.path.exists('Models'):
         os.makedirs('Models')
         
         
    for train, test in skf.split(X, Y):
      start = time.time()
    

      X_train=X[train]
      Y_train=Y[train]
      X_test=X[test]
      Y_test=Y[test] 
    
     #adding image features part
      if image_feats:
          Features_train=Features[train]
          Features_test=Features[test]
          pca = PCA(n_components=4)
          fit=pca.fit(Features_train)
          Features_train=pca.transform(Features_train)
          Features_test=pca.transform(Features_test)
          X_train=np.concatenate((X_train, Features_train), axis=1)
          X_test=np.concatenate((X_test, Features_test), axis=1)
      
      
      #feats=fs.SelectFeatures(X_train,Y_train,thresholds[2],True)
      X_train_logit=X_train
      #X_train=X_train[:,feats]
      X_test_logit=X_test
      #X_test=X_test[:,feats]
      
      scaler = preprocessing.StandardScaler().fit(X_train)
      X_train_svm=scaler.transform(X_train)
      X_test_svm=scaler.transform(X_test)
      X_train_NN=scaler.transform(X_train)
      X_test_NN=scaler.transform(X_test)
      
      
      #normalizer = preprocessing.Normalizer().fit(X_train)
      #X_train_svm=normalizer.transform(X_train) 
      #X_test_svm=normalizer.transform(X_test) 
      #X_train_NN=normalizer.transform(X_train)
      #X_test_NN=normalizer.transform(X_test)
     
      print("Done Selecting Features")
      if RunSVM: 
              #Paramsvm,vals1=mt.GridSearchSVM(X_train_svm,Y_train,GridSplit)
              Paramsvm,vals1=mt.RandomGridSearchSVM(X_train_svm,Y_train,GridSplit)
              print("Done Fine Tuning")
              if Paramsvm['kernel']=='linear':
                  Testsvm[itera,:],fpr_svm,tpr_svm,Accsvm[itera],specsvm[itera],senssvm[itera]=mt.TestSVM(X_train_svm,Y_train,X_test_svm,Y_test,Paramsvm['kernel'],Paramsvm['C'],0,0,itera)
                  vals1[0]=Paramsvm['kernel']
                  vals1[1]=Paramsvm['C']
              else:
                 if Paramsvm['kernel']=='poly':
                     Testsvm[itera,:],fpr_svm,tpr_svm,Accsvm[itera],specsvm[itera],senssvm[itera]=mt.TestSVM(X_train_svm,Y_train,X_test_svm,Y_test,Paramsvm['kernel'],Paramsvm['C'],Paramsvm['gamma'],Paramsvm['degree'],itera)
                     vals1[0]=Paramsvm['kernel']
                     vals1[1]=Paramsvm['C']
                     vals1[2]=Paramsvm['gamma']
                     vals1[3]=Paramsvm['degree']
                 else:
                     Testsvm[itera,:],fpr_svm,tpr_svm,Accsvm[itera],specsvm[itera],senssvm[itera]=mt.TestSVM(X_train_svm,Y_train,X_test_svm,Y_test,Paramsvm['kernel'],Paramsvm['C'],Paramsvm['gamma'],0,itera)
                     vals1[0]=Paramsvm['kernel']
                     vals1[1]=Paramsvm['C']
                     vals1[2]=Paramsvm['gamma']
              gridsvm.append(vals1)
              print("Done testing SVM",Testsvm[itera,1])
              mean_tprs += interp(mean_fprs, fpr_svm, tpr_svm)
              mean_tprs[0] = 0.0
      if RunLogit: 
                   
                   TestLogit[itera,:],fpr_log,tpr_log,AccLog[itera],speclog[itera],senslog[itera]=mt.TestLogistic(X_train_logit,Y_train,X_test_logit,Y_test,itera)
                   print("Done testing Logit",TestLogit[itera,1])
                   mean_tprlog += interp(mean_fprlog, fpr_log, tpr_log)
                   mean_tprlog[0] = 0.0
      if RunRFC:   
                   #Paramsrfc,vals2=mt.GridSearchRFC(X_train,Y_train,GridSplit)
                   Paramsrfc,vals2=mt.RandomGridSearchRFC(X_train,Y_train,GridSplit)
                   vals2[0]=Paramsrfc['n_estimators']
                   vals2[1]=Paramsrfc['max_features']
                   vals2[2]=Paramsrfc['criterion']
                   gridrfc.append(vals2)
                   print("Done Fine Tuning")
                   TestForest[itera,:],fpr_rf,tpr_rf,Accrfc[itera],specrfc[itera],sensrfc[itera]=mt.TestRFC(X_train,Y_train,X_test,Y_test,Paramsrfc['n_estimators'],Paramsrfc['max_features'],Paramsrfc['criterion'],itera)
                   print("Done testing",TestForest[itera,1])
                   mean_tprr += interp(mean_fprr, fpr_rf, tpr_rf)
                   mean_tprr[0] = 0.0
  
      if RunNN:
                #ParamsNN,vals6=mt.GridSearchNN(X_train,Y_train,GridSplit)
                ParamsNN,vals6=mt.RandomGridSearchNN(X_train_NN,Y_train,GridSplit)
                vals6[0]=ParamsNN['activation']
                vals6[1]=ParamsNN['hidden_layer_sizes']
                vals6[2]=ParamsNN['alpha']
                vals6[3]=ParamsNN['batch_size']
                vals6[4]=ParamsNN['learning_rate_init']
                gridNN.append(vals6)
                print(ParamsNN)
                print("Done Fine Tuning")
                Testnn[itera,:],fpr_nn,tpr_nn,Accnn[itera],specnn[itera],sensnn[itera]=mt.TestNN(X_train_NN,Y_train,X_test_NN,Y_test,ParamsNN['activation'],ParamsNN['hidden_layer_sizes'],ParamsNN['alpha'],
                     ParamsNN['batch_size'],ParamsNN['learning_rate_init'],itera)
                print("Done testing", Testnn[itera,1])      
                mean_tprn += interp(mean_fprn, fpr_nn, tpr_nn)
                mean_tprn[0] = 0.0   
      itera=itera+1
      print("ITERATION = ",itera)
      if RunSVM:
          namefeat='Parameters SVM.txt'
          thefile = open(namefeat, 'a')
          for item in Paramsvm:
              thefile.write(" %s " %(Paramsvm[item] ))  
          thefile.write("\n")    
      if RunRFC:
          namefeat='Parameters RFC.txt'
          thefile = open(namefeat, 'a')
          for item in Paramsrfc:
              thefile.write(" %s " %(Paramsrfc[item] )) 
          thefile.write("\n")    
      if RunNN:
          namefeat='Parameters NN.txt'
          thefile = open(namefeat, 'a')
          for item in ParamsNN:
              thefile.write(" %s " %(ParamsNN[item] )) 
          thefile.write("\n")    
              
    print()      
    print("------------- RESULTS ------------") 
    
    f,ax=plt.subplots(figsize=(10,10))
 
    lw=2
    if RunSVM:
      mean_tprs /= skf.get_n_splits(X, Y)
      mean_tprs[-1] = 1.0
      mean_auc_svm = auc(mean_fprs, mean_tprs)
      ax.plot(mean_fprs, mean_tprs, color='darkorange',lw=lw,marker='.', label='SVM (area = %0.2f)' % mean_auc_svm)
      ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    if RunRFC:
      mean_tprr /= skf.get_n_splits(X, Y)
      mean_tprr[-1] = 1.0
      mean_auc_rfc = auc(mean_fprr, mean_tprr)
      ax.plot(mean_fprr, mean_tprr, color='darkblue',lw=lw, linestyle=':',marker='v', label='RFC (area = %0.2f)' % mean_auc_rfc)
      ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    if RunLogit:
      mean_tprlog /= skf.get_n_splits(X, Y)
      mean_tprlog[-1] = 1.0
      mean_auc_log = auc(mean_fprlog, mean_tprlog)
      ax.plot(mean_fprlog, mean_tprlog, color='darkgreen',lw=lw,marker='x',linestyle='-.', label='Logistic (area = %0.2f)' % mean_auc_log)
      ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    if RunNN:
      mean_tprn /= skf.get_n_splits(X, Y)
      mean_tprn[-1] = 1.0 
      mean_auc_nn = auc(mean_fprn, mean_tprn)
      ax.plot(mean_fprn, mean_tprn, color='black',lw=lw,marker='+', label='NN (area = %0.2f)' % mean_auc_nn)
      ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
     
    #plt.plot(ff, tt, color='black',lw=lw,marker='+', label='NN (area = %0.2f)' % mean_auc_log)
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #plt.show
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    
    #plt.axis([0,1,0,1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Average ROC from ML approach ')
    ax.legend(loc="lower right")
    
    #ax.show() 
    fig = ax.get_figure()
    fig.savefig('test.pdf', format='pdf')
    end = time.time()  
     
    namefeat='Feats'+str(itera)+'.txt'
    thefile = open('Sensitivity.txt', 'a')
    
    
     
    if RunSVM: 
      val1=sum(senssvm)/splits
      val2=sum(specsvm)/splits
      val3=sum(Accsvm)/splits
      val5=mean_auc_svm
      val5_1=np.mean(Testsvm[:,1])
      std5_1=np.std(Testsvm[:,1])
    #  val6=sum(briersvm)/splits
      std1=np.std(senssvm)
      std2=np.std(specsvm)
      std3=np.std(Accsvm)
    #  std4=np.std(f1ssvm)
      
     # std6=np.std(briersvm)
      thefile.write("SVM \n")
      thefile.write("Average sensitivity %f and std %f \n" % (val1,std1))
      thefile.write("Average specificity %f and std %f \n" % (val2,std2))
      thefile.write("Average Accuracy %f and std %f \n" % (val3,std3))
     # thefile.write("Average F1-score %f and std %f \n" % (val4,std4))
      thefile.write("Average AUC %f \n" % (val5))
      thefile.write("Average AUC From test VAr %f and AUC-std %f \n" % (val5_1,std5_1))
     # thefile.write("Average Brier %f and Brier-std %f \n" % (val6,std6))
      np.save('fpr-svm.npy',mean_fprs)
      np.save('tpr-svm.npy',mean_tprs)
      mt.Write_ROC(Testsvm,'svm')
               
      
    if RunRFC: 
      val1=sum(sensrfc)/splits
      val2=sum(specrfc)/splits
      val3=sum(Accrfc)/splits
      #val4=sum(f1srfc)/splits
      val5= mean_auc_rfc
      val5_1=np.mean(TestForest[:,1])
      std5_1=np.std(TestForest[:,1])
      std1=np.std(sensrfc)
      std2=np.std(specrfc)
      std3=np.std(Accrfc)
      #std4=np.std(f1srfc)
      thefile.write("RFC \n")
      thefile.write("Average sensitivity %f and std %f \n" % (val1,std1))
      thefile.write("Average specificity %f and std %f \n" % (val2,std2))
      thefile.write("Average Accuracy %f and std %f \n" % (val3,std3))
      #thefile.write("Average F1-score %f and std %f \n" % (val4,std4))
      thefile.write("Average AUC %f and AUC-std \n" % (val5))
      thefile.write("Average AUC From test VAr %f and AUC-std %f \n" % (val5_1,std5_1))
      mt.Write_ROC(TestForest,'RFC')
      np.save('fpr-rfc.npy',mean_fprr)
      np.save('tpr-rfc.npy',mean_tprr)
      
    if RunLogit: 
      val1=sum(senslog)/splits
      val2=sum(speclog)/splits
      val3=sum(AccLog)/splits
      #val4=sum(f1srfc)/splits
      val5= mean_auc_log
      val5_1=np.mean(TestLogit[:,1])
      std5_1=np.std(TestLogit[:,1])
      std1=np.std(senslog)
      std2=np.std(speclog)
      std3=np.std(AccLog)
      #std4=np.std(f1srfc)
      thefile.write("Logit \n")
      thefile.write("Average sensitivity %f and std %f \n" % (val1,std1))
      thefile.write("Average specificity %f and std %f \n" % (val2,std2))
      thefile.write("Average Accuracy %f and std %f \n" % (val3,std3))
      #thefile.write("Average F1-score %f and std %f \n" % (val4,std4))
      thefile.write("Average AUC %f and AUC-std \n" % (val5))
      thefile.write("Average AUC From test VAr %f and AUC-std %f \n" % (val5_1,std5_1))
      mt.Write_ROC(TestLogit,'Logit')
      np.save('fpr-log.npy',mean_fprlog)
      np.save('tpr-log.npy',mean_tprlog)
      
    if RunNN: 
      val1=sum(sensnn)/splits
      val2=sum(specnn)/splits
      val3=sum(Accnn)/splits
     # val4=sum(f1snn)/splits
      val5=mean_auc_nn
      val5_1=np.mean(Testnn[:,1])
      std5_1=np.std(Testnn[:,1])
      std1=np.std(sensnn)
      std2=np.std(specnn)
      std3=np.std(Accnn)
     # std4=np.std(f1snn) 

      thefile.write("NN \n")
      thefile.write("Average sensitivity %f and std %f \n" % (val1,std1))
      thefile.write("Average specificity %f and std %f \n" % (val2,std2))
      thefile.write("Average Accuracy %f and std %f \n" % (val3,std3))
      #thefile.write("Average F1-score %f and std %f \n" % (val4,std4))
      thefile.write("Average AUC %f and AUC-std \n" % (val5))
      thefile.write("Average AUC From test VAr %f and AUC-std %f \n" % (val5_1,std5_1))
      mt.Write_ROC(Testnn,'NN')
      np.save('fpr-nn.npy',mean_fprn)
      np.save('tpr-nn.npy',mean_tprn)
    print("Total time to process: ",end - start)
     #print("sensitivity" ,sensit)
     #print("specificity" ,specif)
     #print("Average sensitivity" ,sum(sensit)/splits)
     #print("Average specificity" ,sum(specif)/splits)
     #print("Average Accuracy" ,sum(Acc)/splits)
     #print("Average Balanced Accuracy" ,sum(BalAcc)/splits)
     #print("Average F1 Score" ,sum(f1s)/splits)
     #print(ts)
    thefile.close() 
    plt.show()
    
    