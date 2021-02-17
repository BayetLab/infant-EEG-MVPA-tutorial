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

from SVM_decode import decode_within_SVM







DataPath      = '../data/'

Datafile      = 'Adults_included.mat'

ParData = loadmat(DataPath+Datafile)

parforArg          = float("inf")   # 0 = no parfor; Inf = parfor

SaveAll            = True

params_decoding = {}

# Classification
params_decoding['function']         = 'decode_within_SVM'
params_decoding['timetime']        = False  # compute time-time generalization (false: only compute the diagonal such that time_test=time_train)
params_decoding['num_permutations'] = 200
params_decoding['L']                = 4     # Number of folds for pseudo-averaging/k-fold

# Data selection
params_decoding['min_cond_rep_per_subj'] = 1      # NaN/0/1 -> all
params_decoding['Epoch_analysis']        = [-50, 500]   # ms

#Extract data name from filename defined above 
params_decoding['DataName'] = Datafile[:len(Datafile)-4]

params_decoding['Date'] = date.today().strftime('%m.%d.%Y')

now = datetime.datetime.now().strftime("%H.%M.%S")

#print(params_decoding)

# select data for analysis, if relevant
Labels = ParData['Y']
if ~np.isnan(params_decoding['min_cond_rep_per_subj']):
    for s in np.unique(ParData['S']):
        for cond in np.unique(Labels):
            if sum(((Labels==cond) & (ParData['S']==s))[0])<params_decoding['min_cond_rep_per_subj']:
                Labels[Labels==cond & ParData['S']==s]=0


# Filter by times/epochs we want to look at
t = ParData['times']

frames = (t>=params_decoding['Epoch_analysis'][0]) & (t<=params_decoding['Epoch_analysis'][1])

selected_epochs = ~np.isnan(Labels)


times = t[frames]

x=ParData['X'][:,frames[0],:]
x=x[:,:,selected_epochs[0]]

labels=Labels[selected_epochs]
s=ParData['S'][selected_epochs]

# Do classification

if params_decoding['function'] == 'decode_within_SVM':
    results= decode_within_SVM(x, labels, s, params_decoding, parforArg,times)
        
#else
    #alternate classification pipelines go here
     

# Save results

if SaveAll:
    if params_decoding['timetime'] ==True:
        timetime_case='_timetime'
    else:
        timetime_case =''
        
    out = 'Results_'+ params_decoding['DataName']+'_'+ params_decoding['function']+ timetime_case
    
    out = out+'_'+params_decoding['Date']+'_'+now+'.mat'
    
    results['out'] = out
    results['results']['out'] = out


    scipy.io.savemat('../Results/'+out, results, do_compression=True)

    