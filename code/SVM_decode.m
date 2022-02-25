
%% Within-subject decoding routine for adult EEG data
% Adapted from pseudocode provided by Radoslaw Cichy (Cichy et al. 2014)

# Input: X = data structure containing channel data 
# Y = corresponding trial condition labels 
# S = corresponding participant numbers 
# params = some decoding parameters 
# parforArg = boolean, run in parallel or sequentially 

function [GroupDA, params, nreps] = SVM_decode(X, Y, S, params, parforArg )

%% Parameters and path
% Parameters
ncond       = length(unique(Y));
conds       = unique(Y);
subjects    = unique(S);
nchan       = size(X,1);
nt          = size(X,2); % a.k.a. num. samples
L           = params.L;
nsubj       = length(unique(S));
nperms      = params.num_permutations;
disp(['...Decoding ',num2str(ncond), ' conditions on data from ',num2str(nsubj),' participants and ',num2str(nchan),' channels'])

% Initialize group results matrices
if ~params.timetime
    GroupDA = NaN([nsubj, nt, ncond, ncond]);
else
    GroupDA = NaN([nsubj, nt, nt, ncond, ncond]);
end
GroupA  = NaN([nsubj, nt, ncond, ncond, nchan]);
nreps = NaN([nsubj ncond]);
for j=1:nsubj; nreps(j,:) = arrayfun(@(i) sum((Y==conds(i))&(S==subjects(j))),1:ncond); end



% flexible bin size to deal with varying number of
% trials/condition/participant
% using algo from https://www.geeksforgeeks.org/split-the-number-into-n-parts-such-that-difference-between-the-smallest-and-the-largest-part-is-minimum/
zp=L*ones(size(nreps)) - mod(nreps,L);

K=repmat(floor(nreps./L), [1 1 L]);

for k = 1:L
    for j = 1:nsubj
        for i = 1:ncond
            if k > zp(j,i)
                K(j,i, k) = K(j,i,k)+1;
            end
        end
    end
end


for i = 1:length(subjects)
    tic
    
    % Prepare subject data
    disp(['....Starting participant ',num2str(i),'/', num2str(nsubj)])
    x=X(:,:,S==subjects(i)); % nchan x frames x ntrials
    y=Y(S==subjects(i)); % ntrials
    
    % Fill data matrix D   
    D= {[]};
    for j = 1:ncond
        D{j} = x(:,:,y==conds(j));%flexible length accomodates varying numbers of repetitions per condition
    end

    % Initialize individual results matrices
    if ~params.timetime
        DA = NaN(nperms,nt,ncond,ncond);
    else
        DAtt = NaN(nperms,nt,nt, ncond,ncond);
    end
    
    
    % Initialize waitbar
    ParallelQueue=parallel.pool.DataQueue;
    hwait = waitbar(0, ['Please wait, working on participant ',num2str(i),'/', num2str(nsubj),'...']);
    afterEach(ParallelQueue, @nUpdateWaitbar);
    progress_p = 1;

   parfor (perm = 1:nperms, parforArg)%set parforarg to 0 to do a simple for loop instead of parfor

        send(ParallelQueue, perm);
        
        %% Form L pseudotrials per condition based on random permutation of trial order within each condition
        
        % Randomly re-order repetitions
        permutedD= {[]};
        for j = 1:ncond
            permutedD{j} = D{j}(:,:,randperm(size(D{j},3)));%trial order is randomized separately in each condition, with flexible length to accomodate varying numbers of repetitions per condition
        end
        
    
        % Binning data into L pseudo trials:
        pseudo_trialD = NaN(ncond, nchan , nt, L);
        trial_selector_start = NaN([ncond, L]);
        trial_selector_end = NaN([ncond, L]);
        for j = 1:ncond
            for step = 1:L
                
                if step == 1
                    trial_selector_start(j,step) = 1;
                else
                    trial_selector_start(j,step) = 1 + sum(K(i,j,1:(step-1)));
                end
                trial_selector_end(j,step) = sum(K(i,j,1:step));
                 
                pseudo_trialD(j,:,:,step) = nanmean(permutedD{j}(:,:,trial_selector_start(j,step) : trial_selector_end(j,step)), 3);
                
                
            end
        end

        % Check for empty bins, for peace of mind
        if sum(isnan(pseudo_trialD(:))) ~= 0
            error('Warning: At least one NaN pseudotrial generated! Something is off here -- maybe fewer than L trials in some conditions?')
        end
        
        %% Loop through all pair-wise combinations of conditions
        if ~params.timetime %#ok<PFBNS>
            DAperm=NaN(nt, ncond, ncond);
        else
            DAperm=NaN(nt, nt, ncond, ncond);
        end
       
        
        for condA = 1:ncond
            for condB = (condA+1):ncond % upper triangle
                
                % Loop through all times (time points are independent)
                for time_point_train = 1:nt
                    
                    %% Train
                    % Train data
                    % L-1 pseudo trials go to testing set, the Lth to training set
                    % size of train set must be m by n where m is number of
                    % instances and n number of features
                    MEEG_training_data=[squeeze(pseudo_trialD(condA,:, time_point_train, 1:end-1)) , ...
                        squeeze(pseudo_trialD(condB, :, time_point_train, 1:end-1))];%chan x (2* (L-1))
                    % Class labels (1-A or 2-B) - size must be m by 1
                    
                    
                    
                    labels_train = [ones(1,L-1) 2*ones(1,L-1)];
                    
                    % for peace of mind
                    if (sum(isnan(MEEG_training_data))) > 0
                        error('some training data is empty! aborting...')
                    end
                    
                    % SVM classification with libsvm
                    % (https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
                    %(see README in matlab folder of the libsvm toolbox)
                    model = svmtrain(labels_train', MEEG_training_data' ,'-s 0 -t 0 -q');  %#ok<SVMTRAIN> -s 0 C-SVM, -t 0 linear, -q no output
                    

                    %% Test
                    if ~params.timetime
                        test_times= time_point_train;
                    else
                        test_times=1:nt;
                    end
                    for time_point_test = test_times
                        
                        % Test set
                        MEEG_testing_data=[squeeze(pseudo_trialD(condA, :, time_point_test, end))' , ...
                            squeeze(pseudo_trialD(condB, :, time_point_test,end))'];%chan*2                    
           
                        % Class labels (1-A or 2-B) - size must be m by 1
                        labels_test  = [1 2];
                        %again, for extra peace of mind
                        if (sum(isnan(MEEG_testing_data))) > 0
                            error('some test data is empty! aborting...')
                        end
                        
                        % Predict
                        [~, accuracy, ~] = svmpredict(labels_test', MEEG_testing_data' , model, '-q');
                        if ~params.timetime
                            DAperm(time_point_train,condA,condB)=accuracy(1);
                        else
                            DAperm(time_point_train,time_point_test,condA,condB)=accuracy(1);
                        end
                        
                    end
                    
                end
                
            end
        end
        
        %% Append permutation result to individual results matrix
        % DA updated outside of loops to comply with classification restrictions (see http://blogs.mathworks.com/loren/2009/10/02/using-parfor-loops-getting-up-and-running/#12)
        if ~params.timetime
            DA(perm,:,:,:)=DAperm;
        else
            DAtt(perm,:,:,:,:)=DAperm;
        end
       
        
  
  
   end
   

    %% average over permutations and append to group results matrix
    if ~params.timetime
        DA_end=squeeze(nanmean(DA,1));
        GroupDA(i,:,:,:)= DA_end;% nsubj x  nt x ncond x ncond
    else
        DA_end=squeeze(nanmean(DAtt,1));
        GroupDA(i,:,:,:,:)= DA_end;% nsubj x ntx nt x ncond x ncond
    end

    
    toc

    close(hwait)
    
    
    
end
    function nUpdateWaitbar(~)
        waitbar(progress_p/nperms, hwait);
        progress_p = progress_p + 1;
    end
end
