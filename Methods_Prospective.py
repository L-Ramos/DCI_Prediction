
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.externals import joblib                                  
from sklearn import svm

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import brier_score_loss
import random as rand

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import ElasticNetCV
from scipy.stats import expon
from scipy.stats import randint as sp_randint
import time
import iari as imp
import warnings

if __name__ == '__main__':
 warnings.filterwarnings("ignore", category=DeprecationWarning)

def Write_ROC(Test,method):
    namefeat='ROC-'+method+'.txt'

    thefile = open(namefeat, 'w')
    for i in range(0,Test.shape[0]):
        thefile.write('%f,\n'%(Test[i,1]))
    thefile.close()  

def RandomGridSearchNN(X,Y,splits):
    scores = ['roc_auc']
    start = time.time()  
    
    tuned_parameters = {
    'activation': (['relu','logistic','tanh']),
    'hidden_layer_sizes':       ([[50,25],[60,30],[60,40,20],[50,30,10],[70,40,20],[70,30],[80,50,30],[80,60,30,10]]), 
    'alpha':     ([0.1, 0.01, 0.001, 0.0001]) ,
    'batch_size':         [ 64, 128],
    'learning_rate_init':    [0.01, 0.05, 0.001, 0.005,0.0001],
#    'class_weight': ['balanced'],
    #'max_iter':     [-1],
    'random_state': [None],}
    print("NN Grid Search")
    mlp = MLPClassifier(max_iter=5000) 
    clf = RandomizedSearchCV(mlp, tuned_parameters, cv= splits, scoring='%s' % scores[0],n_jobs=-1)
        
    clf.fit(X, Y)
         
    end = time.time()
    print("Total time to process: ",end - start)
    return(clf.best_params_,list(clf.best_params_))
    
    
def RandomGridSearchSVM(X,Y,splits):
        
    #tuned_parameters = [{'C': [0.1, 0.01, 0.001, 1, 10, 100], 'kernel': ['linear']},
    #{'C': [0.1, 0.01, 0.001, 1, 10, 100], 'gamma': [ 0.001, 0.0001], 'kernel': ['rbf']},
    # {'C': [0.1, 0.01, 0.001, 1, 10, 100], 'gamma': [ 0.001, 0.0001], 'kernel': ['poly'],'degree':[1,2,3,4]},]
    start = time.time()  
    
    tuned_parameters = {
    'C':            ([0.1, 0.01, 0.001, 1, 10, 100]),
    'kernel':       ['linear', 'rbf','poly'],                   # precomputed,'poly', 'sigmoid'
    'degree':       ([1,2,3,4,5,6]),
    'gamma':         [1, 0.1, 0.01, 0.001, 0.0001],
#    'coef0':        np.arange( 0.0, 10.0+0.0, 0.1 ).tolist(),
    'probability':  [False],
#    'tol':          np.arange( 0.001, 0.01+0.001, 0.001 ).tolist(),
#'cache_size':   [2000],
    'class_weight': ['balanced'],
    'verbose':      [False],
   # 'max_iter':     [-1],
    'random_state': [None],
    }
    
    #{'C': expon(scale=100), 'gamma': expon(scale=.1),
  #'kernel': ['rbf','linear','poly'],degree 'class_weight':['balanced', None]}
  

    scores = ['roc_auc']

   
    w='balanced'
        
   
    print("SVM Grid Search")
    clf =  RandomizedSearchCV(SVC(C=1,class_weight=w), tuned_parameters, cv=splits,
                       scoring='%s' % scores[0],n_jobs=-1)
        
    clf.fit(X, Y)


    end = time.time()
    print("Total time to process: ",end - start)
  
    return(clf.best_params_,list(clf.best_params_.values()))
    
    
    
def RandomGridSearchRFC(X,Y,splits):
        
    
    start = time.time()  
    
    tuned_parameters = {
    'n_estimators': ([100,200,400,600,800]),
    'max_features': (['auto', 'sqrt', 'log2']),                   # precomputed,'poly', 'sigmoid'
    'max_depth':    ([10,20,30]),
    'criterion':    (['gini', 'entropy']),
    'min_samples_split':  [2,3,4,5],
    'min_samples_leaf':   sp_randint(1, 11),
    'class_weight': ['balanced'],
    }
    
    #{'C': expon(scale=100), 'gamma': expon(scale=.1),
  #'kernel': ['rbf','linear','poly'],degree 'class_weight':['balanced', None]}
  

    scores = ['roc_auc']

   
    w='balanced'
        
    rfc = RandomForestClassifier(n_estimators=25, oob_score = True,class_weight=w)   
    
    print("RFC Grid Search")
    clf =  RandomizedSearchCV(rfc, tuned_parameters, cv=splits,
                       scoring='%s' % scores[0],n_jobs=-7)
        
    clf.fit(X, Y)


    end = time.time()
    print("Total time to process: ",end - start)
  
    return(clf.best_params_,list(clf.best_params_.values()))







"""
GridSearch

"""

def GridSearchSVM(X,Y,splits):
    
    
    tuned_parameters = [{'C': [0.1, 0.01, 0.001, 1, 10, 100], 'kernel': ['linear']},
    {'C': [0.1, 0.01, 0.001, 1, 10, 100], 'gamma': [ 0.001, 0.0001], 'kernel': ['rbf']},
     {'C': [0.1, 0.01, 0.001, 1, 10, 100], 'gamma': [ 0.001, 0.0001], 'kernel': ['poly'],'degree':[1,2,3,4]},
    ]
                
                                   
    scores = ['roc_auc']

   
    w='balanced'
        
   
    print("SVM Grid Search")
    clf = GridSearchCV(SVC(C=1,class_weight=w), tuned_parameters, cv=splits,
                       scoring='%s' % scores[0],n_jobs=-1)
        
    clf.fit(X, Y)


  
    return(clf.best_params_,list(clf.best_params_.values()))
        
def GridSearchRFC(X,Y,splits):
   
                                   
    scores = ['roc_auc']
   
    w='balanced'

    print("RFC Grid Search")
    rfc = RandomForestClassifier(max_features= 'sqrt' ,n_estimators=50, oob_score = True,class_weight=w) 

    param_grid = { 'n_estimators': [50,100,200,400,800],'max_features': ['auto', 'sqrt', 'log2'],'criterion':['gini', 'entropy']}
        
    clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= splits, scoring='%s' % scores[0],n_jobs=-1)
        
    clf.fit(X, Y)

   
    return(clf.best_params_,list(clf.best_params_.values()))
   

       
def GridSearchNN(X,Y,splits):
                                   
    scores = ['roc_auc']
  

   # param_grid = {'activation':['relu','logistic','tanh'], 'hidden_layer_sizes': [[10],[20],[50],[10,10],[10,20],[10,50],[10,10,10],
   #                                               [10,10,20],[10,20,20],[10,20,50]],'alpha': [0.1, 0.01, 0.001, 0.0001],'batch_size': [16,32],'learning_rate_init': [0.01, 0.05, 0.001, 0.005]}
    param_grid = {'activation':['relu','logistic','tanh'], 'hidden_layer_sizes': [[10],[20],[30],[10,30],[30,10],[10,10],[30,30]]
                               ,'alpha': [0.1, 0.01, 0.001, 0.0001],'batch_size': [16,32],'learning_rate_init': [0.01, 0.05, 0.001, 0.005]}
    print("NN Grid Search")
    mlp = MLPClassifier(max_iter=5000) 
    clf = GridSearchCV(estimator=mlp, param_grid=param_grid, cv= splits, scoring='%s' % scores[0],n_jobs=-1)
        
    clf.fit(X, Y)
         
    return(clf.best_params_,list(clf.best_params_.values()))
"""
Testing
"""
def TestSVM(X_train,Y_train,X_test,Y_test,kernel,C,gamma,deg,itera):
    result=np.zeros((1,2))
    w='balanced'
    

    if kernel=='linear':
              clf = svm.SVC(C=C,kernel=kernel,class_weight=w)
    else:
              if kernel=='poly':
                 clf = svm.SVC(C=C,kernel=kernel,gamma=gamma,class_weight=w,degree=deg)
              else:
                 clf = svm.SVC(C=C,kernel=kernel,gamma=gamma,class_weight=w)
    
     
    clf.fit(X_train,Y_train)
            
    predictions=clf.predict(X_test)
    decisions = clf.decision_function(X_test)
    
   
    aucD=roc_auc_score(Y_test,decisions)
    auc=roc_auc_score(Y_test,predictions)
    
    name=('Models/SVM'+str(itera)+'.pkl')
    joblib.dump(clf,name)
    
    fpr_svm, tpr_svm, t = roc_curve(Y_test, decisions) 
    
    conf_m=confusion_matrix(Y_test, predictions)
    acc=accuracy_score(Y_test, predictions)
    sensitivity=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
    specificity=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
           
    result[0,0]=auc
    result[0,1]=aucD


    return(result,fpr_svm,tpr_svm,acc,specificity,sensitivity)
    
def TestLogistic(X_train,Y_train,X_test,Y_test,itera):
    
    result=np.zeros((1,2))
    Cvals=[0.1, 0.01, 0.001, 1, 10, 100]#find best parameters for model
    # Solvers="newton-cg", "lbfgs", "sag", "liblinear"
    clf = LogisticRegressionCV(Cs=Cvals,solver=  "liblinear",class_weight='balanced')

     
    clf.fit(X_train,Y_train)
            
    predictions=clf.predict(X_test)
    probas = clf.predict_proba(X_test)[:, 1]
    
   
    aucD=roc_auc_score(Y_test,probas)
    auc=roc_auc_score(Y_test,predictions)
    
    name=('Models/LGT'+str(itera)+'.pkl')
    joblib.dump(clf,name)
    
    fpr_svm, tpr_svm, _ = roc_curve(Y_test, probas) 
    
    conf_m=confusion_matrix(Y_test, predictions)
    acc=accuracy_score(Y_test, predictions)
    sensitivity=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
    specificity=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
           
    result[0,0]=auc
    result[0,1]=aucD


    return(result,fpr_svm,tpr_svm,acc,specificity,sensitivity)
            
            
def TestRFC(X_train,Y_train,X_test,Y_test,n_estim,max_feat,crit,itera):
    result=np.zeros((1,2))
    
    w='balanced'
     
    clf = RandomForestClassifier(max_features=max_feat,n_estimators=n_estim, oob_score = True,class_weight=w,criterion=crit)
           
    clf.fit(X_train,Y_train)
            
    predictions=clf.predict(X_test)
    probas = clf.predict_proba(X_test)[:, 1]
          
   
    auc2=roc_auc_score(Y_test,probas)
    auc=roc_auc_score(Y_test,predictions)
    fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)  
        
    conf_m=confusion_matrix(Y_test, predictions)
    acc=accuracy_score(Y_test, predictions)
    sensitivity=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
    specificity=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
           
    name=('Models/RFC'+str(itera)+'.pkl')
    joblib.dump(clf,name)
    result[0,0]=auc
    result[0,1]=auc2

    return(result,fpr_rf,tpr_rf,acc,specificity,sensitivity)

        
    
def TestNN(X_train,Y_train,X_test,Y_test,act,hid,alpha,batch,learn,itera):
    result=np.zeros((1,2))
   
          
    nn=MLPClassifier(activation=act,hidden_layer_sizes=hid,alpha=alpha,
                         batch_size=batch,learning_rate_init=learn)
            
    nn = nn.fit(X_train, Y_train)
    probas = nn.predict_proba(X_test)[:, 1]
   
            
    predictions=nn.predict(X_test)
           
    auc2=roc_auc_score(Y_test,probas)
    auc=roc_auc_score(Y_test,predictions)
       
    conf_m=confusion_matrix(Y_test, predictions)
    acc=accuracy_score(Y_test, predictions)
    sensitivity=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
    specificity=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
           
    fpr_nn, tpr_nn, _ = roc_curve(Y_test, probas) 
    name=('Models/NN'+str(itera)+'.pkl')
    joblib.dump(nn,name)
    result[0,0]=auc
    result[0,1]=auc2
    
    

    return(result,fpr_nn,tpr_nn,acc,specificity,sensitivity)
    
def PrintResults(Test,grid,feats,TotalFeatures,name):   
    
     pospred=np.argmax(Test[:,0])
     posprob=np.argmax(Test[:,1])
     
     namefeat='write\Results-'+name+'.txt'
     thefile = open(namefeat, 'a')
     thefile.write("Positions of Prediction: %d and Probability: %d " %(pospred,posprob))
     thefile.write(" \n")
     thefile.write("Test values \n")
     s=Test.shape[0]
     for i in range(0,s):
           thefile.write(" %f \n" % Test[i,1] )
     thefile.write("\n")
     
     for item in grid[(TotalFeatures-posprob)-1]:
        thefile.write(" %s " %(item ))  
     thefile.write("\n")   
     thefile.write(str(feats[posprob,:]))
     print("Parameters p\Probability:",grid[(TotalFeatures-posprob)-1])
     print("Positions of Prediction",pospred,Test[pospred,0],"and Probability:",posprob,Test[posprob,1])
     print("Parameters Best Predictions:",grid[(TotalFeatures-pospred)-1])
     print("Parameters p\Probability:",grid[(TotalFeatures-posprob)-1])
     print("Features Predictions:",feats[pospred,:])
     print("Features Probabilities:",feats[posprob,:])
     print(Test)
     thefile.close()
               
     
def Read_Image_Data(image_feats):

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
    return(n,Y)
    
  
def SensitivityAndSpecificity(thresholds,probas,Y):
    
    sensit=np.zeros(thresholds.shape[0])
    specif=np.zeros(thresholds.shape[0])
    BalAcc=np.zeros(thresholds.shape[0])
    Acc=np.zeros(thresholds.shape[0])    
    f1s=np.zeros(thresholds.shape[0]) 
    TPs=np.zeros(thresholds.shape[0])   
    TNs=np.zeros(thresholds.shape[0])   
    binprob=np.zeros(probas.shape[0])
    brier=np.zeros(probas.shape[0])
    for i in range(1,thresholds.shape[0]):
        pp=(probas>thresholds[i])
        tp=0
        fn=0
        tn=0;
        fp=0;
        for j in range(0,len(pp)):
            if pp[j]==True:
                binprob[j]=1
            else:
                binprob[j]=0    
        for j in range(0,len(pp)):
            if binprob[j]==1:
                if binprob[j]==Y[j]    :
                    tp=tp+1
        for j in range(0,len(pp)):
            if Y[j]==1:
                if binprob[j]==0    :
                    fn=fn+1

        for j in range(0,len(pp)):
            if binprob[j]==0:
                if binprob[j]==Y[j]    :
                    tn=tn+1            
    
        for j in range(0,len(pp)):
            if Y[j]==0:
                if binprob[j]==1    :
                    fp=fp+1   
        #sensit[i]=recall_score(Y, binprob, labels=None, pos_label=1, average='binary', sample_weight=None)
        TPs[i]=tp
        TNs[i]=tn
        sensit[i]=tp/(tp+fn)
        specif[i]=tn/(tn+fp)
        BalAcc[i]=(sensit[i]+ specif[i])/2
        Acc[i]=(tp+tn)/(tp+fp+fn+tn)
        f1s[i]=f1_score(Y,binprob,average='weighted')
        brier[i]=brier_score_loss(Y, binprob)
    #pos1=np.argmax(best)
    #pos1=np.argmax(f1s)
    #pos1=np.argmax(TPs+TNs)
    pos1=np.argmax(Acc)
    return(sensit[pos1],specif[pos1],BalAcc[pos1],Acc[pos1],thresholds[pos1],f1s[pos1],brier[pos1])
    
def BalanceData(X,Y):
    Ytest=Y
    Xtest=X
    total=0
    n=Ytest.shape[0]
    val1=int(2*sum(Y))
    Xtestfinal=np.zeros((val1,X.shape[1]))
    Ytestfinal=np.zeros((val1,1))
    for i in range(0,n): 
        if Ytest[i]==1:
            Xtestfinal[total,:]=Xtest[i,:]
            Ytestfinal[total,:]=Ytest[i]
            total=total+1
    i=0
    n=total        
    while (i<n):
        r_num=rand.randrange(0,n-1)
        if Ytest[r_num]==0 and total<(2*sum(Y)):
            Xtestfinal[total,:]=Xtest[r_num,:]
            Ytestfinal[total,:]=0
            total=total+1
            i=i+1
        #print(Ytest[r_num],total,i,n,r_num)
    return(Xtestfinal,Ytestfinal)

def Connect_Image_Features(pat_names,path_image_data):
            
        Path_Reference_dci =path_image_data+"\\Numbers.txt"
        Path_Reference_nodci =path_image_data+"\\NumbersNoDci.txt"
        
        filedci = open(Path_Reference_dci, 'r')
        filenodci = open(Path_Reference_nodci, 'r')
        found = np.zeros(pat_names.shape[0])
        features=np.zeros((pat_names.shape[0],2592))
        
        for i in range(0,4):
            #path_dci="E:\\RenanCodeCNTKandPatches\\cntk\\Patches\\Patches102\\Prospective\\FoldExtract\\Selection00" + str(i) + "DCI.txt"
            path_dci="E:\\DCI Prediction\\Data\\Image_data\\Selection00"+str(i)+"DCI.txt"
            #path_nodci="E:\\RenanCodeCNTKandPatches\\cntk\\Patches\\Patches102\\Prospective\\FoldExtract\\Selection00" + str(i) + "noDCI.txt"
            path_nodci="E:\\DCI Prediction\\Data\\Image_data\\Selection00"+str(i)+"noDCI.txt"
            reader_dci = open(path_dci, 'r')
            reader_nodci = open(path_nodci, 'r')
            read_features=np.loadtxt('E:\\DCI Prediction\\Data\\Image_data\\autoencoderfeaturesTestFold'+str(i)+'.txt');
            
            dcipat = np.zeros(30,dtype='int32')
            nodcipat = np.zeros(70,dtype='int32')
            txt_dci=reader_dci.readline()
            txt_dci=reader_dci.readline()
            txt_nodci=reader_nodci.readline()
            txt_nodci=reader_nodci.readline()
            k1=0
            k2=0
            
            while "Training" not in txt_dci:
                    dcipat[k1] = int(txt_dci)
                    txt_dci = reader_dci.readline()
                    k1=k1+1
            while "Training" not in txt_nodci:
                    nodcipat[k2] = int(txt_nodci)
                    txt_nodci = reader_nodci.readline()
                    k2=k2+1

            #file_name=filedci.readline()
            #blank_pos=file_name.index(" ")
            #print(file_name)
            #number_ref_dci=int(file_name[blank_pos+1:len(file_name)-1])
            init_pos_dci=0   
            init_pos_nodci=0   
            
            for f in range(0,read_features.shape[0]):                 
                    if(init_pos_dci<k1):
                      
                            file_name=filedci.readline()
                            blank_pos=file_name.index(" ")
                            number_ref_dci=int(file_name[blank_pos+1:len(file_name)])
                            #print(file_name)
                            #print(number_ref_dci,dcipat[init_pos_dci])
                            
                            patpos=file_name.index("set2")
                            dotpos = file_name.index("RCT")
                            patient_number=int(file_name[patpos+5:dotpos-1])
                            
                            for j in range(0,pat_names.shape[0]):
                                if pat_names[j]==patient_number:
                                    features[j]=read_features[init_pos_dci]
                                    found[j]=found[j]+1
                                    print("Found at:",patient_number,pat_names[j],init_pos_dci,j)
                                    pat_names[j]=0
                            init_pos_dci = init_pos_dci+1
                    else:
                            file_name=filenodci.readline()
                            blank_pos=file_name.index(" ")
                            number_ref_dci=int(file_name[blank_pos+1:len(file_name)])
                           # print(file_name)
                           # print(number_ref_dci,nodcipat[init_pos_nodci])
                           
                            patpos=file_name.index("set2")
                            dotpos = file_name.index("RCT")
                            patient_number=int(file_name[patpos+5:dotpos-1])
                            
                            for j in range(0,pat_names.shape[0]):
                                if pat_names[j]==patient_number:
                                    features[j]=read_features[init_pos_dci]
                                    found[j]=found[j]+1
                                    print("Found at:",patient_number,pat_names[j],init_pos_dci,j)                           
                                    pat_names[j]=0
                            init_pos_dci = init_pos_dci+1 
            print("Fold",i)
        return(features)
            
        