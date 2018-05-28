# -*- coding: utf-8 -*-
"""
This code tests several combinations for number of features reduced with PCa

@author: laramos
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


n_test0=np.loadtxt('E:\\RenanCodeCNTKandPatches\\cntk\\Features\\NP3\\F\\autoencoderfeaturesTestFold0.txt');
l_test0=np.loadtxt('E:\\RenanCodeCNTKandPatches\\cntk\\Features\\NP3\\F\\autoencoderfeaturesTestFold0L.txt');
n_test1=np.loadtxt('E:\\RenanCodeCNTKandPatches\\cntk\\Features\\NP3\\F\\autoencoderfeaturesTestFold1.txt');
l_test1=np.loadtxt('E:\\RenanCodeCNTKandPatches\\cntk\\Features\\NP3\\F\\autoencoderfeaturesTestFold1L.txt');
n_test2=np.loadtxt('E:\\RenanCodeCNTKandPatches\\cntk\\Features\\NP3\\F\\autoencoderfeaturesTestFold2.txt');
l_test2=np.loadtxt('E:\\RenanCodeCNTKandPatches\\cntk\\Features\\NP3\\F\\autoencoderfeaturesTestFold2L.txt');
n_test3=np.loadtxt('E:\\RenanCodeCNTKandPatches\\cntk\\Features\\NP3\\F\\autoencoderfeaturesTestFold3.txt');
l_test3=np.loadtxt('E:\\RenanCodeCNTKandPatches\\cntk\\Features\\NP3\\F\\autoencoderfeaturesTestFold3L.txt');


n=np.concatenate((n_test0,n_test1,n_test2,n_test3), axis=0)
l=np.concatenate(( l_test0,l_test1,l_test2,l_test3), axis=0)

n=np.array(n,dtype='float32')
l=np.array(l,dtype='float32')

Y=np.zeros(l.shape[0])


for i in range(0,l.shape[0]):
    if l[i,0]==1:
        Y[i]=1

splits=10
skf = ShuffleSplit(n_splits=splits, test_size=0.25,random_state=1)


kpca = KernelPCA(kernel="linear" )
n_train = kpca.fit(n)
n_test_write = kpca.transform(n_test1)
n_train_write = kpca.transform(n)
print(n_train_write.shape)

auc=np.zeros(10)
aucp=np.zeros(10)
aucpt=np.zeros(10)
auct_sum=np.zeros(200)
auct_std=np.zeros(200)
auc_sum=np.zeros(200)
auc_std=np.zeros(200)
l=0

c=0
i=2592
while i>4:    
    l=0
    for train, test in skf.split(n, Y): 
        pca = PCA(n_components=int(i))
        fit=pca.fit(n[train])
        n_train=pca.transform(n[train])
        n_test=pca.transform(n[test])
               
        w='balanced'
        #clf = svm.SVC(C=0.01,kernel='linear',probability=True,class_weight='balanced')
        Cs=[0.1, 0.01, 0.001, 1, 10, 100]
        clf=svm.SVC(C=100,kernel='linear',probability=False,class_weight=w)
        #clf = LogisticRegressionCV(Cs=Cs,class_weight='balanced',solver='liblinear')        
        #clf=RandomForestClassifier(max_features='log2',n_estimators=50, oob_score = True,criterion='gini')
        clf.fit(n_train,Y[train])
        predictions=clf.predict(n_test)       
        auc[l] = roc_auc_score(Y[test],predictions)
        #probas = clf.predict_proba(n_test)[:, 1]
        probas = clf.decision_function(n_test)
        #probast = clf.predict_proba(n_train)[:, 1]
        probast = clf.decision_function(n_train)
        aucp[l] = roc_auc_score(Y[test],probas)
        predictions=np.array(predictions,dtype='int32')
        aucpt[l]=roc_auc_score(Y[train],probast)
        if (aucpt[l]<0.5):
            aucpt[l]=1-aucpt[l]
        if (aucp[l]<0.5):
            aucp[l]=1-aucp[l]
        
        l=l+1
    print("Total Features = ",i)    
    auct_sum[c]=np.sum(aucpt)/10
    auct_std[c]=np.std(aucpt)/10
    auc_sum[c]=np.sum(aucp)/10
    auc_std[c]=np.std(aucp)/10
    print("AUc T = ",auct_sum[c])
    print("AUc P= ",auc_sum[c])
    i=i/2
    c=c+1
  #print("Average AUC Training",sum(aucpt)/100," ",np.std(aucpt)) 
  #print("Average AUC Probas",sum(aucp)/100," ",np.std(aucp))  


