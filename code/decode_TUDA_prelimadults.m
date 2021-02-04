% Based on run_TUDA from https://github.com/vidaurre/CC2018 

% It needs the HMM-MAR toolbox, which can be forked from here: 
% https://github.com/OHBA-analysis/HMM-MAR
clear all 

addpath(genpath(['/Users/bayet/Dropbox (Personal)/Data/EAGER/paper2/repo-private/infant-decode-private/non-shared-toolboxes/HMM-MAR-master']))
load('/Users/bayet/Dropbox (Personal)/Data/EAGER/paper2/repo-private/infant-decode-private/data/Adults_included.mat')

options = struct();
options.K = 3; % 4 number of decoders 
options.pca = 0; 
options.DirichletDiag = 1000;
options.detrend = 0; % already filtered
options.onpower = 0 ; % run on raw signal
options.standardise = 1; % standardize data
options.parallel_trials = 1; % trials are aligned (to stimulus presentation)
options.tol = 1e-5;
options.cyc = 100; % a bit quicker than by default
options.initcyc = 10;
options.initrep = 3;
options.verbose = 1;
options.useParallel = 0;%added to skip the problematic chucking of the data that causes a bug
options.classifier = 'LDA';%LDA runs into trouble with obslike
%issue here = 'Undefined function or variable 'WishTrace' 
% Data (here try out with one subcase, then will loop)
subj=1;% within subj 
catA=1; catB = 2; % pairwise

selected_trials = (S == subj)&((Y == catA)|(Y == catB));
data=X(:,:,selected_trials);%all channels and times
data = permute(data,[2,3,1]); %permute to be times x trials x sensors
y = Y(selected_trials);
y(y == catA) = -1;
y(y == catB) = 1;

if sum(y == -1) ~= sum(y == 1) %unbalanced!
    reorder_trials = randperm(size(data,2));
    data = data(:,reorder_trials,:);
    y = y(reorder_trials);
    
    dataA = data(:,y == -1,:);
    dataB = data(:,y == 1,:);
    yA = y(y == -1);
    yB = y(y == 1);
    targetntrial = min(sum(y == -1), sum(y == 1)); 
    dataA = dataA(:,1:targetntrial,:);
    dataB = dataB(:,1:targetntrial,:);
    yA = yA(1:targetntrial);
    yB = yB(1:targetntrial);
    y_balanced = [yA, yB];
    data_balanced = cat(2, dataA, dataB);
    
    reorder_trials = randperm(size(data_balanced,2));
    data_balanced = data_balanced(:,reorder_trials,:);
    y_balanced = y_balanced(reorder_trials);
    
else
    data_balanced = data;
    y_balanced = y;
end

N = size(data_balanced,2); 
T = repmat(size(data_balanced,1),[size(data_balanced,2) 1]);

% Train tuda
[tuda,Gamma,GammaInit] = tudatrain(data_balanced,y_balanced,T,options);
% Encoding model
encmodel = tudaencoding(data_balanced,y_balanced,T,options,Gamma);

