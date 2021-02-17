from sklearn import svm
from sklearn import metrics
from scipy.io import loadmat
import scipy.io
from datetime import date
import datetime
import numpy as np
import os.path
from os import path
import numpy.matlib
import math
import multiprocessing
from joblib import Parallel, delayed


def parallel_decode_fun(perm,params,ncond,nchan,nt,D,L,K,i):



    permutedD= []
    for j in range(0,ncond):
        permutedD.append(D[j][:,:,np.random.permutation(D[j].shape[2])])    #trial order is randomized separately in each condition, with flexible length to accomodate varying numbers of repetitions per condition

    # Binning data into L pseudo trials:

    pseudo_trialD = np.empty([ncond, nchan , nt, L])
    pseudo_trialD[:] = np.NaN

    trial_selector_start = np.empty([ncond, L])
    trial_selector_end = np.empty([ncond, L])
    for j in range(0,ncond):
        for step in range(0,L):
            if step == 0:
                trial_selector_start[j,step] = 0
            else:
                trial_selector_start[j,step] = 1+np.sum(K[i,j,0:step])

            trial_selector_end[j,step] = np.sum(K[i,j,0:step+1])

            pseudo_trialD[j,:,:,step] = np.nanmean(permutedD[j][:,:,int(trial_selector_start[j,step]) : int(trial_selector_end[j,step])], axis=2)



    # Check for empty bins, for piece of mind

    if (np.any(np.isnan(pseudo_trialD[:]))):
        return("Warning: At least one NaN pseudotrial generated! Something is off here -- maybe fewer than L trials in some conditions?")




    # Loop through all pair-wise combinations of conditions
    if ~params['timetime']:
        DAperm=np.empty([nt, ncond, ncond])
    else:
        DAperm=np.empty([nt, nt, ncond, ncond])
    DAperm[:] = np.NaN

    Aperm=np.empty([nt, ncond, ncond,nchan]) 
    Aperm[:] = np.NaN    

    for condA in range(0,ncond):
        for condB in range(condA+1,ncond): #upper triangle

            # Loop through all times (time points are independent)
            for time_point_train in range(0,nt):


                # Train
                # Train data
                # L-1 pseudo trials go to testing set, the Lth to training set
                # size of train set must be m by n where m is number of
                # instances and n number of features

                  # must be shape (m, n_features)

                lA = np.shape(pseudo_trialD[condA, 0:, time_point_train, 0:])[1]
                lB = np.shape(pseudo_trialD[condB, 0:, time_point_train, 0:])[1]

                MEEG_training_data =np.concatenate(( np.squeeze(pseudo_trialD[condA,:, time_point_train, 0:lA-1]),
                                    np.squeeze(pseudo_trialD[condB, :, time_point_train, 0:lB-1])),1).T


                # Class labels (1-A or 2-B) - size must be m by 1
                labels_train = np.concatenate((np.ones([1,L-1]).astype(int),
                                               2*np.ones([1,L-1]).astype(int)), 1)[0]


                # for peace of mind

                if (np.any(np.isnan(MEEG_training_data))):
                    return('some training data is empty! aborting...')


                # SVM classification with sklearn
                # (https://scikit-learn.org/stable/modules/svm.html)


                clf = svm.SVC(kernel = 'linear')
                model = clf.fit(MEEG_training_data,labels_train)

		
                # TODO
                # Commented out to save memory
                # Activation pattern (Haufe et al. 2014)
                w = clf.coef_
                latent_s = np.matmul(MEEG_training_data, w.T)  
                a = np.squeeze(np.divide(np.matmul(np.cov(MEEG_training_data.T, bias=True), w.T), np.cov(latent_s.T, bias = True)))

                Aperm[time_point_train,condA,condB,:] = a



                # Test
                if ~params['timetime']:
                    test_times= [time_point_train]
                else:
                    test_times=list(range(0,nt))




                for time_point_test in test_times:

                    # Test set
                    lA = np.shape(pseudo_trialD[condA, 0:, time_point_test, 0:])[1]
                    lB = np.shape(pseudo_trialD[condB, 0:, time_point_test, 0:])[1]



                    MEEG_testing_data=np.concatenate((np.squeeze(pseudo_trialD[condA, :, time_point_test,lA-1]).reshape(1,-1),
                                                      np.squeeze(pseudo_trialD[condB, :, time_point_test,lB-1]).reshape(1,-1)))



                    # Class labels (1-A or 2-B) - size must be m by 1

                    labels_test  = [1, 2]

                    #again, for extra peace of mind
                    if (np.any(np.isnan(MEEG_testing_data))):
                        return('some testing data is empty! aborting...')



                    predictions = clf.predict(MEEG_testing_data)


                    accuracy = metrics.accuracy_score(labels_test, predictions, normalize = False)
                    accuracy = accuracy*100


                    if ~params['timetime']:
                        DAperm[time_point_train,condA,condB]=accuracy
                    else:
                        DAperm[time_point_train,time_point_test,condA,condB]=accuracy




    return(DAperm,Aperm, perm)






















# parameters: X = data matrix , Y = label array ,
#    S = selected epochs , params = dictionary of parameters, parforArg = Inf, times=time window of interest

def decode_within_SVM(X, Y, S, params, parforArg, times):

    # Parameters

    ncond       = len(np.unique(Y))
    conds       = np.unique(Y)
    subjects    = np.unique(S)
    nchan       = np.shape(X)[0]
    nt          = np.shape(X)[1] # a.k.a. num. samples
    L           = params['L']
    nsubj       = len(np.unique(S))
    nperms      = params['num_permutations']

    print('...Decoding '+str(ncond)+ ' conditions on data from '+str(nsubj)+' participants and '+str(nchan)+' channels')

    # Initialize group results matrices


    if ~params['timetime']:
        GroupDA = np.empty([nsubj, nt, ncond, ncond])

    else:
        GroupDA = np.empty([nsubj, nt, nt, ncond, ncond])
    GroupDA[:] = np.NaN

    GroupA  = np.empty([nsubj, nt, ncond, ncond, nchan])
    GroupA[:] = np.NaN

    nreps = np.empty([nsubj, ncond])

    for j in range(0,nsubj):
        for k in range(0, ncond):
            nreps[j,k] = sum((Y==conds[k])&(S==subjects[j]))

    # flexible bin size to deal with varying number of
    # trials/condition/participant
    # using algo from https://www.geeksforgeeks.org/split-the-number-into-n-parts-such-that-difference-between-the-smallest-and-the-largest-part-is-minimum/

    zp=L*np.ones([nsubj, ncond]) - (nreps%L)


    K = np.repeat(np.floor_divide(nreps,L)[:, :, np.newaxis], L, axis=2)



    for k in range(0,L):
        for j in range(0,nsubj):
            for i in range(0,ncond):
                if k > zp[j,i]:
                    K[j,i, k] = K[j,i,k]+1;



    for i in range(0,len(subjects)):
        print('....Starting participant '+str(i+1)+'/'+str(nsubj))

        x=X[:,:,S==subjects[i]]
        y=Y[S==subjects[i]]

        D=[]
        for j in range(0,ncond):
            D.append(x[:,:,y==conds[j]])

        DA = np.empty([nperms,nt,ncond,ncond])
        DA[:] = np.NaN


       # Commented out to save memory
        A = np.empty([nperms, nt, ncond, ncond,nchan])
        A[:] = np.NaN


        num_cores = multiprocessing.cpu_count()


        # Run parallel loop

        results=Parallel(n_jobs=num_cores)(delayed(parallel_decode_fun)(perm,params,ncond,nchan,nt,D,L,K,i) for perm in range(0,nperms))

        for res in results:

            if ~params['timetime']:
                DA[res[2],:,:,:]=res[0]
            else:
                DA[res[2],:,:,:,:]=res[0]
            A[res[2],:,:,:,:] = res[1]




        # average over permutations and append to group results matrix

        if ~params['timetime']:
            DA_end=np.squeeze(np.nanmean(DA,0))
            GroupDA[i,:,:,:]= DA_end   # nsubj x  nt x ncond x ncond
        else:
            DA_end=np.squeeze(np.nanmean(DA,0))
            GroupDA[i,:,:,:,:]= DA_end   # nsubj x nt x nt x ncond x ncond


# Commented out to save memory
        A_end=np.squeeze(np.nanmean(A,0))
        GroupA[i,:,:,:,:]= A_end   # 1 x nt x ncond x ncond x nchan




    out = {}
    out['results'] = {}
    out['results']['DA'] = GroupDA
    out['results']['params_decoding'] = params
    out['results']['nreps'] = nreps
    out['results']['times'] = times
    out['results']['A'] = GroupA



    return(out)




