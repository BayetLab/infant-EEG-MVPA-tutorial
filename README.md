1) An overview of the various directories/scripts included in the repository, and what each script is used for; 2) At least a brief, step-by-step guide/walk-through describing which scripts to run, in what order, and the resulting output/interpretation of each step.


# infant-EEG-MVPA-tutorial
Tutorial toolbox for MVPA decoding of EEG data 
Classification is implemented in Python and Matlab
Euclidean distance based decoding is implemented in python only

# Python Workflow 

Input data should be in the form of a .mat file containing four structs      
-> S : array of size 1 x total number of trials; containing participant numbers corresponding to all trials    
-> X : array of size number of channels x number of time points x number of trials; contains preprocessed channel voltage data at each time point for each trial     
-> Y : array of size 1 x total number of trials; containing condition labels corresponding to all trials       
-> times : array of size 1 x number of time points; contains all time points in the epoch of interest     

### Arguments   
  -p --path                Specify path to input EEG data folder, relative or absolute       
  -f --file                Name of input EEG data file    
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
      In command line:
        python run_MVPA_decoding.py -p 'C:/user/pathtodata/data/' -f 'Infants_all.mat' -par 1 -s 1 -d 'SVM_decode' -n 200 -k 4 -ts -50 -te 500 -tt False     
      or: 
        python run_MVPA_decoding.py (arguments set manually in script, not recommended)
      
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

    from scipy.io import loadmat
    
    # provide path to folder where results are stored
    
    DataPath = 'Results/'
    
    # define the name of the output file you want to look at
    
    fname='Results_Adults_all_decode_within_SVM_10.06.2021_12.49.15.mat'
    
    # Extract the decoding accuracy results 
    
    data = loadmat(DataPath+fname)['results'][0,0]
    DA = data['DA']


# Matlab Workflow

Input data should be in the form of a .mat file containing four structs      
-> S : array of size 1 x total number of trials; containing participant numbers corresponding to all trials    
-> X : array of size number of channels x number of time points x number of trials; contains preprocessed channel voltage data at each time point for each trial     
-> Y : array of size 1 x total number of trials; containing condition labels corresponding to all trials       
-> times : array of size 1 x number of time points; contains all time points in the epoch of interest   

### Arguments   
-> All arguments are defined manually at the begining of the script 

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

    from scipy.io import loadmat
    
    # provide path to folder where results are stored
    
    DataPath = 'Results/'
    
    # define the name of the output file you want to look at
    
    fname='Results_Adults_all_decode_within_SVM_10.06.2021_12.49.15.mat'
    
    # Extract the decoding accuracy results 
    
    data = loadmat(DataPath+fname)['results'][0,0]
    DA = data['DA']
    
    
    
   

