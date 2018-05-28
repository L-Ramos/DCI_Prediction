# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:28:38 2018

@author: laramos
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def Plot_ROC(path,pdf_title):

    fpr_svm=np.load(path+'fpr-svm.npy')
    tpr_svm=np.load(path+'tpr-svm.npy')
    
    fpr_rfc=np.load(path+'fpr-rfc.npy')
    tpr_rfc=np.load(path+'tpr-rfc.npy')
    
    fpr_log=np.load(path+'fpr-log.npy')
    tpr_log=np.load(path+'tpr-log.npy')
    
    fpr_nn=np.load(path+'fpr-nn.npy')
    tpr_nn=np.load(path+'tpr-nn.npy')

    f,ax=plt.subplots(figsize=(10,10))
 
    lw=2

    mean_auc_rfc = auc(fpr_rfc, tpr_rfc)
    mean_auc_rfc = 74
    ax.plot(fpr_rfc, tpr_rfc, color='darkblue',lw=lw, linestyle=':',marker='v', label='RFC (area = %0.2f)' % mean_auc_rfc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    
    mean_auc_svm = auc(fpr_svm, tpr_svm)
    mean_auc_svm = 68
    ax.plot(fpr_svm, tpr_svm, color='darkorange',lw=lw,marker='.', label='SVM (area = %0.2f)' % mean_auc_svm,markersize=10)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    mean_auc_nn = auc(fpr_nn, tpr_nn)
    mean_auc_nn = 66
    ax.plot(fpr_nn, tpr_nn, color='black',lw=lw,marker='x',linestyle='-.', label='NN (area = %0.2f)' % mean_auc_nn)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    mean_auc_log = auc(fpr_log, tpr_log)
    mean_auc_log = 65
    ax.plot(fpr_log, tpr_log, color='darkgreen',lw=lw, label='Logistic (area = %0.2f)' % mean_auc_log,markersize=10)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw)
     
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    
    
    #plt.axis([0,1,0,1])
    ax.set_xlabel('False Positive Rate',fontsize=12)
    ax.set_ylabel('True Positive Rate',fontsize=12)
    #ax.set_title('Average ROC from Machine Learning experiments')
    ax.set_title('Average ROC from Machine Learning and Image Features experiments',fontsize=12)
    ax.legend(loc="lower right")
    
    #ax.show() 
    fig = ax.get_figure()
    fig.savefig(pdf_title, format='pdf')
    #fig.savefig('Clin.pdf', format='pdf')


with_image=True

if with_image:
    path="E:\\PhdRepos\\ClinicalNeuralNetworkSVM\\Prospective\\Results  clinical plus image and ROc values\\"
else:
    path="E:\\PhdRepos\\ClinicalNeuralNetworkSVM\\Prospective\\Results only clinical and ROc values\\"

pdf_title="clinical_and_image.pdf"    
Plot_ROC(path,pdf_title)    

