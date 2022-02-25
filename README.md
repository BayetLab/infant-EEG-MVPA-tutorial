# infant-EEG-MVPA-tutorial
Tutorial toolbox for MVPA decoding of infant EEG data implemented in Python and Matlab

## Python functions
  
  run_MVPA_decoding.py    
  - Runs decoding scripts with given input 

  SVM_decode.py   
  - Decoding with classification accuracy, called by run_MVPA_decoding.py

  Euclidean_decode.py   
  - Decoding with euclidean distance, called by run_MVPA_decoding.py


## Matlab functions   
  
  run_MVPA_decoding.m   
  - Runs decoding scripts with given input

  SVM_decode.m   
  - Decoding with classification accuracy, called by run_MVPA_decoding.m 
   

## Usage: 
      python run_MVPA_decoding.py [arguments]
      or: 
      python run_MVPA_decoding.py (arguments set manually in script, not recommended)

### Arguments:   
  -p --path                Specify path to input EEG data folder, relative or absolute       
  -f --file                Name of input EEG data file    
  -par --parallel          1 to run in parallel (distributed on multiple cores), 0 to run sequentially (default=1)   
  -s --save                1 to save all decoding output, 0 to run without saving (default=1)   
  -d --decode_method       Name of decoding method function ('SVM_decode' or 'Euclidean_decode')   
  -n --nperms              Number of trial permutations for decoding (default=200)   
  -k --nfolds              Number of pseudo-trials to generate for cross validation (default=4)   
  -ts --time_start         Start of epoch (ms)    
  -te --time_end           End of epoch (ms)   
  -tt --time_time          Compute time-time generalization (default=0: time_test=time_train)   

```matlab
run run_MVPA_decoding.m
```
  - All decoding parameters are set manually in script before running
