# Notes about data sets

Structs in results:
- X data, processed and z-scored to baseline (channels x samples x trials)
- Y trial condition (1 x trials)
- S participants (1 x trials)
- times time vector (1 x samples)
- nreps number of trials per condition (participants x conditions)

-> Conditions are coded as integers from 1 to 8

Four sample datasets are provided: 
- Adults_all N=9 all available artifact-free data
- Adults_included N=8 all artifact-free data from participants included in Bayet et al 2020 (i.e., all participants with at least 50% valid trials)
- Infants_all N=21 all available valid data
- Infants_included N=10 all available artifact free data from participants included in Bayet et al 2020 (i.e., all participants with at least 50% valid trials)

If using these data, please cite:
  Ashton, K., Zinszer, B. D., Cichy, R. M., Nelson III, C. A., Aslin, R. N., & Bayet, L. (2022). Time-resolved multivariate pattern analysis of infant EEG data: A practical tutorial. Developmental Cognitive Neuroscience, 101094
  Bayet, L., Zinszer, B. D., Reilly, E., Cataldo, J. K., Pruitt, Z., Cichy, R. M., Nelson III, C. A., & Aslin, R. N. (2020). Temporal dynamics of visual representations in the infant brain. Developmental cognitive neuroscience, 45, 100860 

