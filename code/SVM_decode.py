from sklearn import svm
from sklearn import metrics
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import time
import warnings

#Within-subject decoding routine for EEG data
#Adapted from pseudocode provided by Radoslaw Cichy (Cichy et al. 2014)

def parallel_decode_fun(perm,params,ncond,nchan,nt,D,L,K,i):
    warnings.filterwarnings(action='ignore', message='Mean of empty slice') #ignoring this warning because there will be empty slices, since this is a sparse array, it isn't actually a problem
    permutedD= []
    for j in range(0,ncond):
        permutedD.append(D[j][:,:,np.random.permutation(D[j].shape[2])])    #trial order is randomized separately in each condition, with flexible length to accomodate varying numbers of repetitions per condition


    pseudo_trialD = np.empty([ncond, nchan , nt, L])

    for j in range(0,ncond):
        for step in range(0,L):
            if step == 0:
                trial_selector_start = 0
            else:
                trial_selector_start = np.sum(K[i,j,0:step])

            trial_selector_end = np.sum(K[i,j,0:step+1])

            pseudo_trialD[j,:,:,step] = np.nanmean(permutedD[j][:,:,int(trial_selector_start) : int(trial_selector_end)], axis=2)


    # Check for empty bins, for piece of mind

    if (np.any(np.isnan(pseudo_trialD[:]))):
        return("Warning: At least one NaN pseudotrial generated! Something is off here -- maybe fewer than L trials in some conditions?")


    if ~params['timetime']:
        DAperm=np.empty([nt, ncond, ncond])
    else:
        DAperm=np.empty([nt, nt, ncond, ncond])
    DAperm[:] = np.NaN

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

                MEEG_training_data =np.concatenate(( np.squeeze(pseudo_trialD[condA,:, time_point_train, 0:-1]),
                                    np.squeeze(pseudo_trialD[condB, :, time_point_train, 0:-1])),1).T

              
                # Class labels (1-A or 2-B) - size must be m by 1
                labels_train = np.concatenate((np.ones([1,L-1]).astype(int),
                                               2*np.ones([1,L-1]).astype(int)), 1)[0]


                # for peace of mind

                if (np.any(np.isnan(MEEG_training_data))):
                    return('some training data is empty! aborting...')


                # SVM classification with sklearn


                clf = svm.SVC(kernel = 'linear', decision_function_shape='ovo', gamma='auto')
                model = clf.fit(MEEG_training_data,labels_train)


                # Test
                if ~params['timetime']:
                    test_times= [time_point_train]
                else:
                    test_times=list(range(0,nt))



         
                for time_point_test in test_times:
                    

                    MEEG_testing_data=np.concatenate((np.squeeze(pseudo_trialD[condA, :, time_point_test,-1]).reshape(1,-1),
                                                      np.squeeze(pseudo_trialD[condB, :, time_point_test,-1]).reshape(1,-1)))

                        
                    # Class labels (1-A or 2-B) - size must be m by 1

                    labels_test  = [1, 2]

                    #again, for extra peace of mind
                    if (np.any(np.isnan(MEEG_testing_data))):
                        return('some testing data is empty! aborting...')



                    predictions = clf.predict(MEEG_testing_data)

                    accuracy = metrics.accuracy_score(labels_test, predictions, normalize = True)
                    accuracy = accuracy*100
                    
                    if ~params['timetime']:
                        DAperm[time_point_train,condA,condB]=accuracy
                    else:
                        DAperm[time_point_train,time_point_test,condA,condB]=accuracy
                    

 
    return(DAperm, perm)


def decode_within_SVM(X, Y, S, params, parforArg, times):

    warnings.filterwarnings(action='ignore', message='Mean of empty slice') #ignoring this warning because there will be empty slices, since this is a sparse array, it isn't actually a problem
    
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

    GroupDA = []
    nreps = np.array([[sum((Y==k)&(S==j)) for k in conds] for j in subjects])
     

    
    # flexible bin size to deal with varying number of
    # trials/condition/participant
    # using algo from https://www.geeksforgeeks.org/split-the-number-into-n-parts-such-that-difference-between-the-smallest-and-the-largest-part-is-minimum/

    K = []
    for sub in range(0,nsubj):
        cond = []
        for c in range(0,ncond):
            bins = np.floor_divide(nreps[sub,c],L) * np.ones([L])
            count = nreps[sub,c]
            for b in range(0,L):
                if(sum(bins)<count):
                    bins[b] = bins[b]+1
            cond.append(bins)
        K.append(cond)
    


    K=np.array(K)
              
           
                    
    for i in range(0,len(subjects)):

        x=X[:,:,S==subjects[i]]
        y=Y[S==subjects[i]]
        yhist=np.array([len(y[y==x]) for x in Y])        

        if np.any(yhist<L):
            print("Participant "+str(i+1)+" does not have enough trials in at least 1 condition and has been skipped")
            continue

        print('....Starting participant '+str(i+1)+'/'+str(nsubj))


        start=time.time()


        D=[]
        for j in range(0,ncond):
            D.append(x[:,:,y==conds[j]])

        DA=[]


        param=[[perm,params,ncond,nchan,nt,D,L,K,i]for perm in range(0,nperms)]



        num_cores = multiprocessing.cpu_count()



        if parforArg:
            results=Parallel(n_jobs=num_cores)(delayed(parallel_decode_fun)(perm,params,ncond,nchan,nt,D,L,K,i) for perm in range(0,nperms))
        else:
            results=[parallel_decode_fun(perm,params,ncond,nchan,nt,D,L,K,i) for perm in range(0,nperms)]
            

        end=time.time()
        print(str((end-start)/60)+" minutes")
        
        for res in results:
            DA.append(np.asarray(res[0]))

        DA=np.asarray(DA)
        
        # average over permutations and append to group results matrix

        DA_end=np.squeeze(np.nanmean(DA,0))
        GroupDA.append(DA_end)   # nsubj x  nt x ncond x ncond
      
    out = {}
    out['results'] = {}
    out['results']['DA'] = GroupDA
    out['results']['params_decoding'] = params
    out['results']['nreps'] = nreps
    out['results']['times'] = times
    

    return(out)
