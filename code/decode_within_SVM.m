function [GroupDA, params, nreps, GroupA] = decode_within_SVM(X, Y, S, params, parforArg )
% function [GroupDA, params, nreps, GroupA] = decode_within_SVM(X, Y, S, params, parforArg )
%% Within-subject decoding routine for adult EEG data
% Adapted from pseudocode provided by Radoslaw Cichy (Cichy et al. 2014)

% TODOS
% Consider not including computation of the "activation" variable - unclear how useful here
% Update way to deal with unequal numbers of trials to be less nonsensical - e.g. use cell not array

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

for i = 1:length(subjects)
    tic
    
    % Prepare subject data
    disp(['....Starting participant ',num2str(i),'/', num2str(nsubj)])
    x=X(:,:,S==subjects(i)); % nchan x frames x ntrials
    y=Y(S==subjects(i)); % ntrials
    
    % Fill data matrix D
    D = NaN([ncond, nchan, nt, max(nreps(i,:))]); % ncond x nchan x  nt  x  nrep
    for j = 1:ncond
        D(j,:,:,1:nreps(i,j)) = x(:,:,y==conds(j));
    end
    K = ceil(size(D,4)/L); % size of each of the L pseudo-averaging bins. for best results nreps should multiples L.
    
    % Initialize individual results matrices
    if ~params.timetime
        DA = NaN(nperms,nt,ncond,ncond);
    else
        DAtt = NaN(nperms,nt,nt, ncond,ncond);
    end
    A = NaN(nperms, nt, ncond, ncond,nchan);
    
    % Initialize waitbar
    ParallelQueue=parallel.pool.DataQueue;
    hwait = waitbar(0, ['Please wait, working on participant ',num2str(i),'/', num2str(nsubj),'...']);
    afterEach(ParallelQueue, @nUpdateWaitbar);
    progress_p = 1;
    
    parfor (perm = 1:nperms, parforArg)%set parforarg to 0 to do a simple for loop instead of parfor

        send(ParallelQueue, perm);
        
        %% Form L pseudotrials per condition based on random permutation of trial order within each condition
        % TODO: update all this to better deal with unequal numbers of trials. Use cell not array.
        allset = 0;
        maxit=500;
        it=1;
        while allset==0
            
           
            % Randomly re-order repetitions
            permutedD=NaN(size(D));% ncond x nchan x  nt  x  nrep
            for j=1:ncond
                permutedD(j,:,:,:) = D(j,:,:,randperm(size(D,4))); % /!\ trial order is randomized separately in each condition - important for excluding random draws that have empty pseudotrials in some conditions without this affecting how other conditions are permuted 
            end
            
            % Binning data into L pseudo trials:
            % ....first K trials go to bin 1, the next K to bin 2, etc. to form L=nrep/K pseudo trials
            pseudo_trialD = NaN(ncond, nchan , nt, L);
            for step = 1:L
                if step<L
                    trial_selector  = (1+(step-1)*K):(K+(step-1)*K); %select trials to be averaged
                    pseudo_trialD(:,:,:,step) = nanmean(permutedD(:,:,:,trial_selector), 4);
                else
                    pseudo_trialD(:,:,:,step) = nanmean(permutedD(:,:,:,(1+(step-1)*K):end), 4);
                end
            end
            
            % Check for empty bins
            if sum(isnan(pseudo_trialD(:))) == 0
                allset=1;
            else
                if it<maxit
                    warning('Warning: at least one NaN pseudotrial generated! Trying new permuted trial orders...')
                    it=it+1;
                else
                    allset=1; %#ok<NASGU>
                    error('Warning: Maxit reached with at least one NaN pseudotrial generated! :(( Increase maxit manually (line 70) and rerun')
                end
            end
        end
        
        %% Loop through all pair-wise combinations of conditions
        if ~params.timetime %#ok<PFBNS>
            DAperm=NaN(nt, ncond, ncond);
        else
            DAperm=NaN(nt, nt, ncond, ncond);
        end
        Aperm=NaN(nt, ncond, ncond,nchan);
        
        for condA = 1:ncond
            for condB = (condA+1):ncond % upper triangle
                
                % Loop through all times (time points are independent)
                for time_point_train = 1:nt
                    
                    %% Train
                    % Train data
                    % L-1 pseudo trials go to testing set, the Lth to training set
                    % size of train set must be m by n where m is number of
                    % instances and n number of features
                    MEEG_training_data=[squeeze(pseudo_trialD(condA,:, time_point_train, 1:end-1)) , squeeze(pseudo_trialD(condB, :, time_point_train, 1:end-1))];%chan x (2* (L-1))
                    % Class labels (1-A or 2-B) - size must be m by 1
                    labels_train = [ones(1,L-1) 2*ones(1,L-1)];
                    % for piece of mind
                    if (sum(isnan(MEEG_training_data))) > 0
                        error('Code is not working to prevent empty train data. :((')
                    end
                    
                    % SVM classification with libsvm
                    % (https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
                    %(see README in matlab folder of the libsvm toolbox)
                    model = svmtrain(labels_train', MEEG_training_data' ,'-s 0 -t 0 -q');  %#ok<SVMTRAIN> -s 0 C-SVM, -t 0 linear, -q no output
                    
                    % Activation pattern (Haufe et al. 2014)
                    w = (model.sv_coef' * full(model.SVs));% http://stackoverflow.com/questions/10131385/matlab-libsvm-how-to-find-the-w-coefficients
                    latent_s = MEEG_training_data' * w';% can also be approximated by labels_train for simplicity. its covariance should be a scalar in this case.
                    a = cov(MEEG_training_data') * w' / cov(latent_s);% Haufe et al. 2014 Neuroimage [Eq.5]
                    Aperm(time_point_train,condA,condB,:)=a;
                    
                    %% Test
                    if ~params.timetime
                        test_times= time_point_train;
                    else
                        test_times=1:nt;
                    end
                    for time_point_test = test_times
                        
                        % Test set
                        MEEG_testing_data=[squeeze(pseudo_trialD(condA, :, time_point_test, end))' , squeeze(pseudo_trialD(condB, :, time_point_test,end))'];%chan*2                    
                        % Class labels (1-A or 2-B) - size must be m by 1
                        labels_test  = [1 2];
                        %for piece of mind
                        if (sum(isnan(MEEG_testing_data))) > 0
                            error('Code is not working to prevent empty train data. :((')
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
        A(perm,:,:,:,:)=Aperm;

    end

    %% average over permutations and append to group results matrix
    if ~params.timetime
        DA_end=squeeze(nanmean(DA,1));
        GroupDA(i,:,:,:)= DA_end;% nsubj x  nt x ncond x ncond
    else
        DA_end=squeeze(nanmean(DAtt,1));
        GroupDA(i,:,:,:,:)= DA_end;% nsubj x ntx nt x ncond x ncond
    end
    A_end=squeeze(nanmean(A,1));
    GroupA(i,:,:,:,:)= A_end;% 1 x nt x ncond x ncond x nchan
    
    toc
    close(hwait)
    
end
    function nUpdateWaitbar(~)
        waitbar(progress_p/nperms, hwait);
        progress_p = progress_p + 1;
    end
end