addpath(genpath('tools/'));

%%%%%%%%%%%%%%%%%%%%%%%
%iccv09: 0 void   1,1 sky  0,2 tree   2,3 road  1,4 grass  1,5 water  1,6 building  2,7 mountain 2,8 foreground
set(0,'RecursionLimit',1000);
params.numLabels = 7; % we never predict 0 (void)
params.numFeat = 128;


%%%%%%%%%%%%%%%%%%%%%%
% model parameters (should be ok, found via CV)
params.numHid = 50;
params.regPTC = 0.0001;
params.regC = params.regPTC;
params.LossPerError = 0.05;

%sigmoid activation function:
params.f = @(x) (1./(1 + exp(-x)));
params.df = @(z) (z .* (1 - z));
params.actFunc = 'sigmoid';

mainDataSet = 'penn-fudan-all';
neighNameStem = ['../data/' mainDataSet '-allNeighborPairs'];

%---------------------------------------------------------------
run('/home/hushell/working/deep/vlfeat-0.9.16/toolbox/vl_setup');

%%% --- default parameter values --- %%%
nlabel = 7;

fprintf('processing the features!!\n');

dataFolder = '/home/hushell/working/deep/RNN/codeAndDataSocherICML2011/data/penn-fudan-all-n1/';
dataFolder = [dataFolder 'allInMatlab/'];

testList = trainList; % recompute training data or testing data
force = 0;
whiten = 1;
recompute = 0;
if (~exist('allData','var') || force)
    if exist('allData','var')
        clear allData;
    end
    
    if recompute == 0
        if isTrainData
            load('../data/penn-fudan-allData-train.mat');
        else
            load('../data/penn-fudan-allData-eval.mat');
        end
    else
        allData{length(testList)} = {};
        for i = 1:length(testList)
            allData{i} = load([dataFolder testList{i}(1:end-4) '.mat']);
        end

        if whiten == 1
            % 'whiten' inputs (each feature separately) to mean 0
            if (~exist('meanAll','var'))
                allFeats = [];
                for i = 1:length(allData)
                    allFeats = [allFeats ; allData{i}.feat2];
                end
                meanAll = mean(allFeats) + 1e-5;
                stdAll  = std(allFeats) + 1e-5;
            else
                neighNameTrain = [neighNameStem '_train.mat'];
                %neighNameTrain = [neighNameStem '_train_tiny.mat'];
                load(neighNameTrain ,'meanAll','stdAll');
            end

            % normalize features
            for i = 1:length(allData)
                featsNow = allData{i}.feat2;
                featsNow = bsxfun(@minus, featsNow, meanAll);
                % Truncate to +/-3 standard deviations and scale to -1 to 1
                pstd = 3 * stdAll;
                featsNow = bsxfun(@max,bsxfun(@min,featsNow,pstd),-pstd);
                featsNow = bsxfun(@times,featsNow,1./pstd);
                if strcmp(params.actFunc,'sigmoid')
                    % Rescale from [-1,1] to [0.1,0.9]
                    featsNow = (featsNow + 1) * 0.4 + 0.1;
                end
                allData{i}.feat2 = featsNow;
            end
        end

        for i = 1:length(allData)
            labelRegs = allData{i}.labels;
            segs = allData{i}.segs2;
            numSegs = max(segs(:));
            segLabels = zeros(numSegs,1);
            for r = 1:numSegs
                segLabels(r) = mode(labelRegs(segs==r));
            end
            allData{i}.segLabels = segLabels;

            % find neighbors!
            adjHigher = getAdjacentSegments(segs);%getAdjacentSegments(segs,1)
            adj = adjHigher|adjHigher';
            allData{i}.adj = adj;

            clear segLabels labelRegs segs numSegs;
        end
    end
end

if (~exist('Wcat','var'))
    load ../output/penn-fudan-all-n1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05.mat
end

verbose = 1;
tot_err = 0;
tot_sp = 0;
tot_err_part = zeros(nlabel, 1);
tot_sp_part = zeros(nlabel, 1);
for i = 1:length(allData)
    % GT
    %gt_splabels = multi_output(allData{i}.segLabels,nlabel);
    gt_splabels = allData{i}.segLabels';
    
    % read superpixel features
    feat = allData{i}.feat2;
    numFeat = size(feat, 1);
    feat = [feat, ones(numFeat, 1)]';

    topCorr=0;
    imgTreeTop = parseImage(topCorr,Wbot,W,Wout,Wcat,allData{i}.adj, ...
        allData{i}.feat2,allData{i}.segLabels,params);
    
    pred = imgTreeTop.nodeCat(1:numFeat);
    
    err = sum(pred(:) ~= gt_splabels(:));
    tot_err = tot_err + err;
    tot_sp = tot_sp + numFeat;
    
    for p = 1:nlabel
        tpred = pred(gt_splabels == p);
        tsplabels = gt_splabels(gt_splabels == p);
        err = sum(tpred(:) ~= tsplabels(:));
        tot_err_part(p) = tot_err_part(p) + err;
        tot_sp_part(p) = tot_sp_part(p) + numel(tpred);
    end
    
    if verbose,
        fprintf('valid: [%d/%d] err: %d/%d, acc = %g\n', ...
            i,length(allData),err,numFeat,100*(1-tot_err/tot_sp));
        colorImgWithLabels(allData{i}.segs2,allData{i}.labels,pred,...
            allData{i}.segLabels, allData{i}.img);
    else
        if ~mod(i,10),
            fprintf('.');
        end
        if ~mod(i,100),
            fprintf('[%d/%d] ',i,length(datanames));
            fprintf('acc = %g\n',100*(1-tot_err/tot_sp));
        end
    end
end
acc = 100*(1-tot_err/tot_sp);
fprintf('\nacc = %g\n', acc);

for p = 1:nlabel
    acc = 100*(1-tot_err_part(p)/tot_sp_part(p));
    fprintf('acc of part %d = %g\n', p, acc);
end