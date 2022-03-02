# infant-EEG-MVPA-tutorial
Tutorial toolbox for MVPA decoding of EEG data    
Classification is implemented in Python and Matlab     
Euclidean distance based decoding implemented in python only     

### Directories
```
.
├── code                       # Matlab and Python decoding scripts  
│   └── Euclidean_decode.py    # Supporting Python script for calculating cross validated Euclidean distance, user does not run 
│   └── SVM_decode.m           # Supporting Matlab script for running classification, user does not run
│   └── SVM_decode.py          # Supporting Python script for running classification, user does not run
│   └── run_MVPA_decoding.m    # Matlab script that the user runs to perform MVPA
│   └── run_MVPA_decoding.py   # Python script that the user runs to perform MVPA
│   └── helpers                # Other supporting scripts and files   
│       └── dir2.m           
├── data                       # Infant and adult processed input data files       
│   └── Adults_all.mat         # All adult data
│   └── Adults_included.mat    # Adult data included in bayet et. al. 2020
│   └── Infants_all.mat        # All infant data
│   └── Infants_included.mat   # Infant data included in bayet et. al. 2020
│   └── README.md
└── ex_accuracyplot.jpg        # Example output from the plotting code described below
└── README.md      
```

If using these data, please cite:

- Ashton, K., Zinszer, B. D., Cichy, R. M., Nelson III, C. A., Aslin, R. N., & Bayet, L. (2022). Time-resolved multivariate pattern analysis of infant EEG data: A practical tutorial. Developmental Cognitive Neuroscience, 101094
- Bayet, L., Zinszer, B. D., Reilly, E., Cataldo, J. K., Pruitt, Z., Cichy, R. M., Nelson III, C. A., & Aslin, R. N. (2020). Temporal dynamics of visual representations in the infant brain. Developmental cognitive neuroscience, 45, 100860

# Python Workflow 

Input data should be in the form of a .mat file containing four structs      
-> S : array of size 1 x total number of trials; containing participant numbers corresponding to all trials    
-> X : array of size number of channels x number of time points x number of trials; contains preprocessed channel voltage data at each time point for each trial     
-> Y : array of size 1 x total number of trials; containing condition labels corresponding to all trials       
-> times : array of size 1 x number of time points; contains all time points in the epoch of interest     

### Parameters   
  -p --path                Specify path to input data folder, relative or absolute       
  -f --file                Name of input data file    
  -par --parallel          1 to run in parallel (distributed on multiple cores), 0 to run sequentially (default=1)   
  -s --save                1 to save all decoding output, 0 to run without saving (default=1)   
  -d --decode_method       Name of decoding method function ('SVM_decode' or 'Euclidean_decode')   
  -n --nperms              Number of trial permutations for decoding (default=200)   
  -k --nfolds              Number of pseudo-trials to generate for cross validation (default=4)   
  -ts --time_start         Start of epoch (ms)    
  -te --time_end           End of epoch (ms)   
  -tt --time_time          Compute time-time generalization (default=False: time_test=time_train)   

-> parameters with defaults do not need to be included unless you want to change the default values   
      
### Example usage
```python
#      In command line:
        python run_MVPA_decoding.py -p 'C:/user/pathtodata/data/' -f 'Infants_all.mat' -par 1 -s 1 -d 'SVM_decode' -n 200 -k 4 -ts -50 -te 500 -tt False     
#      or: 
        python run_MVPA_decoding.py (arguments set manually in script, not recommended)
```      
### Output  
Matlab (.mat) file containing structs ‘out’ and ‘results’    
  - The ‘out’ field contains the string name of the file    
  - The ‘results’ field contains:
    - Decoding accuracy : DA : 4-d double matrix of the resulting decoding accuracies of the shape number of participants x number of timepoints x number of conditions x number of conditions
    - Euclidean distance : E : 4-d double matrix of the resulting euclidean distances of the shape number of participants x number of timepoints x number of conditions x number of conditions
      - only the upper diagonal matrix (elements above the diagonal) will contain numbers, while the diagonal and lower diagonal matrix will contain NaNs (not a number) 
    - params_decoding : structure containing the decoding parameters 
      - function : which decoding function (i.e. SVM_decode)
      - timetime : False if training and testing are performed on the same time points
      - num_permutations : number of permutations, default 200
      - Epoch_analysis : time window
      - DataName : name of data file
      - Date : date results were generated
    - nreps : matrix containing the number of trials completed for each participant in each condition
    - times: a list of all time points 


### Auxiliary python functions 
(user doesn't call these functions)    

SVM_decode.py      
  - Decoding with classification accuracy, called by run_MVPA_decoding.py     


Euclidean_decode.py       
  - Decoding with euclidean distance, called by run_MVPA_decoding.py      

### Parsing output for analysis
```python
from scipy.stats import sem
import matplotlib.pyplot as plt
import numpy as np
import mne.stats as mstats
from scipy.io import loadmat
```
```python

    # provide path to folder where results are stored
    DataPath = 'Results/'
    
    # define the name of the output file
    fname='Results_Adults_all_decode_within_SVM_10.06.2021_12.49.15.mat'
    
    # Extract the decoding accuracy results     
    data = loadmat(DataPath+fname)['results'][0,0]
    DA = data['DA']
```    

### Calculate average classification accuracy over the time series

```python

# number of participants
numParts = np.shape(DA)[0]

# set time window
times = [-50, 500]
# set baseline length 
base = 50 
# get array indices
timeInds = [times[0] + base, times[1] + base]

# calculate average classification accuracy for each participant over all conditions at each time point, flatten condition x condition matrix to look at pairwise accuracy
partAccuracies = np.array([[np.nanmean(np.ndarray.flatten(DA[part,point,:,:])) for point in range(timeInds[0],timeInds[1])] for part in range(np.shape(PyInfDA)[0])])

# calculate the group average classification accuracy over all conditions at each time point, flatten condition x condition matrix to look at pairwise accuracy
groupAccuracy = np.array([np.nanmean(np.ndarray.flatten(DA[:,point,:,:])) for point in range(timeInds[0],timeInds[1])])

```
### Significance of classification accuracy against chance

```python
## Calculate significance of group average classification

b = np.full([numParts,550],50) # What are you testing against: in this case, theoretical chance of 50%

T_obs, clusters, cluster_p_values, H0 = mstats.permutation_cluster_test([partAccuracies,b], tail=1, out_type="mask")

```
### Plot time series classification accuracy
```python
## Plot group accuracy, participant accuracies over time series, and bars indicating where accuracy was significantly above chance

plt.figure(figsize=(20,10))  

# Plot all participant accuracies  
for part in partAccuracies:
    plt.plot(range(times[0],times[1]),part)

# Calculate standard error 
error = [sem(partAccuracies[:,i]) for i in range(timeInds[0],timeInds[1])]

# Plot standard error
plt.fill_between(range(times[0],times[1]), groupAccuracy-error,groupAccuracy+error, alpha=0.3, color="teal")

# Plot group average accuracy
plt.plot(range(times[0],times[1]),groupAccuracy, linewidth=5,color="black", label="Group average accuracy")

# Plot significance bars
plt.subplot
for i_c, c in enumerate(iallclusters):
    c = c[0]
    if iallcluster_p_values[i_c] < 0.05:
        h = plt.axvspan(c.start-base, c.stop-base, 
                        color='black', alpha=0.5,ymin=0.17,ymax=0.19)
    
plt.hlines(50, -50, 500)

plt.ylabel('Classification Accuracy %',fontsize=30)   
plt.xlabel('Time (ms)',fontsize=30)

plt.ylim(35, 90)
plt.xlim(times)

plt.tick_params(axis='x', labelsize=30)  
plt.tick_params(axis='y', labelsize=30)  

plt.legend(framealpha=1, frameon=True, fontsize=20,loc="upper right")

plt.show()

```

# Matlab Workflow

Input data should be in the form of a .mat file containing four structs      
-> S : array of size 1 x total number of trials; containing participant numbers corresponding to all trials    
-> X : array of size number of channels x number of time points x number of trials; contains preprocessed channel voltage data at each time point for each trial     
-> Y : array of size 1 x total number of trials; containing condition labels corresponding to all trials       
-> times : array of size 1 x number of time points; contains all time points in the epoch of interest   

### Parameters   
-> All parameters are defined manually at the begining of the script 

```matlab
DataPath      = '../data';
ToolboxesPath = '../non-shared-toolboxes';
Datafile      = 'Infants_included.mat';

% Run
parforArg          = Inf;   % 0 = no parfor; Inf = parfor
ExitMatlabWhenDone = false; % if running as batch on a cluster
SaveAll            = true;

% Classification
params_decoding.function         = 'SVM_decode';
params_decoding.timetime         = false; % compute time-time generalization (false: only compute the diagonal such that time_test=time_train)
params_decoding.num_permutations = 200;
params_decoding.L                = 4; % Number of folds for pseudo-averaging/k-fold

% Data selection
params_decoding.min_cond_rep_per_subj = NaN;  % NaN/0/1 -> all
params_decoding.Epoch_analysis        = [-50 500];% ms
```
### Additional dependencies
-> Matlab implementation requires the following additional toolbox, to be installed in the folder defined by the ToolboxesPath parameter       
- libsvm : toolbox supporting SVM classification, found here https://www.csie.ntu.edu.tw/~cjlin/libsvm/    

### Example usage

```matlab
run run_MVPA_decoding.m
```
### Auxiliary matlab functions 
(user doesn't call these functions)      
SVM_decode.m       
  - Decoding with classification accuracy, called by run_MVPA_decoding.m  

### Output  
Matlab (.mat) file containing structs ‘out’ and ‘results’    
  - The ‘out’ field contains the string name of the file    
  - The ‘results’ field contains:
    - DA : 4-d double matrix of the resulting decoding accuracies of the shape number of participants x number of timepoints x number of conditions x number of conditions
      - only the upper diagonal matrix (elements above the diagonal) will contain numbers, while the diagonal and lower diagonal matrix will contain NaNs (not a number) 
    - params_decoding : structure containing the decoding parameters 
      - function : which decoding function (i.e. SVM_decode)
      - timetime : False if training and testing are performed on the same time points
      - num_permutations : number of permutations, default 200
      - min_cond_rep_per_subj : minimum trials per subject, default NaN (not a number)
      - Epoch_analysis : time window
      - DataName : name of data file
      - Date : date results were generated
    - nreps : matrix containing the number of trials completed for each participant in each condition
    - times: a list of all time points 


### Parsing output for analysis

    % provide path to folder where results are stored
    DataPath = 'Results/'
    
    % define the name of the output file
    fname ='Results_Adults_all_decode_within_SVM_10.06.2021_12.49.15.mat'
    
    % load results data
    load(fullfile(DataPath,fname(1).name),'results');

 
