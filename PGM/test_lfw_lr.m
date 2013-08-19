run('/home/hushell/working/deep/vlfeat-0.9.16/toolbox/vl_setup');
addpath('model/lr');

%%% --- default parameter values --- %%%
config_lr;
startup_penn_fudan;
nlabel = 7;

fprintf('processing the features!!\n');

dataFolder = '/home/hushell/working/deep/RNN/codeAndDataSocherICML2011/data/penn-fudan-all/';
dataFolder = [dataFolder 'allInMatlab/'];

testList = trainList;
force = 1;
if (~exist('allData','var') || force)
    if exist('allData','var')
        clear allData;
    end
    allData{length(testList)} = {};
    for i = 1:length(testList)
        allData{i} = load([dataFolder testList{i}(1:end-4) '.mat']);
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
        clear segLabels labelRegs segs numSegs;
    end
end

if (~exist('w_lr','var'))
    load('weights/lr_without_norm_feat.mat');
end

verbose = 0;
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

    labelprob = inference_lr(feat, w_lr.nodeWeights);
    [~, pred] = max(labelprob ,[], 1);
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