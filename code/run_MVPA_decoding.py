from scipy.io import loadmat
import scipy.io
from datetime import date
import datetime
import numpy as np
from os import path
import numpy.matlib
from SVM_decode import decode_within_SVM
from Euclidean_decode import decode_euclidean_dist
import warnings

import argparse
import sys

########## INPUT ARGS
# Arguments can be manually changed in the script or set when calling the function in the command line (recommended)
# Some arguments can be left as defults (see README)


pars = argparse.ArgumentParser()

pars.add_argument('-p','--path', help='Path to data' )
pars.add_argument('-f','--file', help='Name of input data file' )
pars.add_argument('-par','--parallel', help='Run in parallel', type=int )
pars.add_argument('-s','--save', help='Save output', type=int )
pars.add_argument('-d','--decode_method', help='Name of decoding function name' )
pars.add_argument('-n','--nperms', help='Number of permutations', type=int )
pars.add_argument('-k','--nfolds', help='Number of cross validation folds', type=int )
pars.add_argument('-ts','--time_start', help='Start of epoch (ms)', type=int )
pars.add_argument('-te','--time_end', help='End of epoch (ms)', type=int )
pars.add_argument('-tt', '--time_time', help='', type=int)

args = pars.parse_args()

params_decoding = {}

if np.any(args != None):
   
    if args.path==None: raise Exception("Please enter a valid path to data file") 
    else: DataPath = args.path

    if args.file==None: raise Exception("Please enter a valid data file name")
    else: Datafile = args.file
    

    ParData = loadmat(DataPath+Datafile)

    if args.parallel==None: parforArg = 1
    else: parforArg = args.parallel

    if args.save==None: SaveAll = True
    else: SaveAll = bool(args.save)
 
    if args.decode_method==None: params_decoding['function'] = 'decode_within_SVM'
    else: params_decoding['function'] = args.decode_method

    if args.nperms==None: params_decoding['num_permutations'] = 200
    else: params_decoding['num_permutations'] = args.nperms

    if args.nfolds==None: params_decoding['L'] = 4
    else: params_decoding['L'] = args.nfolds

    if args.time_time==None:  params_decoding['timetime'] = False 
    else: params_decoding['timetime'] = bool(args.time_time)


    if args.time_start==None:
        st = ParData['times'][0][0]
    else:
        st = args.time_start
    if args.time_end==None: 
        en = ParData['times'][0][-1]+1
    else:
        en = args.time_end       
    params_decoding['Epoch_analysis'] = [ st, en ]
    

# If no command line args are used, defaults are assigned here
# File names and paths can be manually set here

else: 

    DataPath      = '../data/'

    Datafile      = 'Infants_included.mat' # set data file for decoding here

    parforArg          = 1   # 0 = not parallel 1 = parallel
    SaveAll            = True # save output 

    params_decoding = {}

    # Classification
    params_decoding['function']         = 'decode_within_SVM'  #'decode_within_SVM' or 'decode_euclidean_dist'
    params_decoding['timetime']        = False  # compute time-time generalization (false: only compute the diagonal such that time_test=time_train)
    params_decoding['num_permutations'] = 200
    params_decoding['L']                = 4     # Number of folds for pseudo-averaging/k-fold
    # Data selection
    params_decoding['Epoch_analysis']        = [-50, 500]   # ms



#Extract data name from filename defined above

params_decoding['DataName'] = Datafile[:len(Datafile)-4]
params_decoding['Date'] = date.today().strftime('%m.%d.%Y')

now = datetime.datetime.now().strftime("%H.%M.%S")

Labels = ParData['Y']

# Filter by times/epochs we want to look at
t = ParData['times'] # time range in ms 

frames = (t>=params_decoding['Epoch_analysis'][0]) & (t<=params_decoding['Epoch_analysis'][1]) # filter for epoch specified in params

selected_epochs = ~np.isnan(Labels) # only applicable if you're filtering labels by setting them to nan, otherwise Labels 

times = t[frames] 

# Filter for participant data within the selected epoch
x=ParData['X'][:,frames[0],:]
x=x[:,:,selected_epochs[0]]

labels=Labels[selected_epochs]

s=ParData['S'][selected_epochs]

# Do classification

if params_decoding['function'] == 'decode_within_SVM': # return pairwise classification accuracy 
    results= decode_within_SVM(x, labels, s, params_decoding, parforArg,times)
    
elif params_decoding['function'] == 'decode_euclidean_dist': # return euclidean distance between condition response patterns
    results= decode_euclidean_dist(x, labels, s, params_decoding, parforArg,times)
    
else:
    print('Please select a valid decoding method')        
   

# Save results

if SaveAll:
    if params_decoding['timetime']:
        timetime_case='_timetime'
    else:
        timetime_case =''

    out = 'Results_'+ params_decoding['DataName']+'_'+ params_decoding['function']+ timetime_case

    out = out+'_'+params_decoding['Date']+'_'+now+'.mat'

    results['out'] = out
    results['results']['out'] = out

    scipy.io.savemat('../Results/'+out, results, do_compression=True)
