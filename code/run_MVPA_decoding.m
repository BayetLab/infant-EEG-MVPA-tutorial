%% Basic classification pipeline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all  %#ok<CLALL>

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data path and file
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize
if SaveAll
    diary(fullfile(pwd, '../results',['Analysis_log_',datestr(now, 'dd-mmm-yyyy'),'.txt']));
    diary ON
end

% Path
addpath(genpath(ToolboxesPath));
addpath(genpath('helpers'));
load(fullfile(DataPath,Datafile));
[~,params_decoding.DataName] = fileparts(Datafile);
params_decoding.Date = date;

disp(params_decoding)

%% Decoding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select data for analysis, if relevant
Labels=Y;
if ~isnan(params_decoding.min_cond_rep_per_subj)
    for s=unique(S)
        for cond=unique(Labels)
            if sum(Labels==cond & S==s)<params_decoding.min_cond_rep_per_subj
                Labels(Labels==cond & S==s)=NaN;
            end
        end
    end
end
frames = times>=params_decoding.Epoch_analysis(1) & times<=params_decoding.Epoch_analysis(2);
selected_epochs = ~isnan(Labels);
results.times = times(frames);
x=X(:,frames,selected_epochs);
labels=Labels(selected_epochs);
s=S(selected_epochs);

% do classification
if strcmp(params_decoding.function, 'decode_within_SVM')
    [results.DA, results.params_decoding, results.nreps] = SVM_decode(x, labels, s, params_decoding, parforArg);
   % else
    % alternate classification pipelines go here
end



% save
if SaveAll
    if params_decoding.timetime ==true; timetime_case='_timetime'; else, timetime_case =''; end
    out = ['Results_', params_decoding.DataName,'_', params_decoding.function, timetime_case];
    out=[out,'_',date,'_',num2str(round(rem(now,1)*100000))];
    results.out=out;
    if (~exist(fullfile('../results',out),'file')); save(fullfile('../results',out),'out'); end
    save(fullfile('../results',out),'results','-append');
end


%% Wrap up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('done.')
if SaveAll
    diary OFF
end
if ExitMatlabWhenDone
    exit %#ok<UNRCH> %%#ok<MSNU> #ok<UNRCH>
end
