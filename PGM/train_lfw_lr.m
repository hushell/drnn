%%% train by logistic regression
clear, close all

%%% --- default parameter values --- %%%
config_lr;
startup_penn_fudan;
nlabel = 7;

fprintf('\nstart LR training\n\n');
fprintf('processing the features!!\n');

dataFolder = '/home/hushell/working/deep/RNN/codeAndDataSocherICML2011/data/penn-fudan-all/';
dataFolder = [dataFolder 'allInMatlab/'];

if ~exist('allData','var')
    allData{length(trainList)} = {};
    for i = 1:length(trainList)
        allData{i} = load([dataFolder trainList{i}(1:end-4) '.mat']);
    end
end

% % 'whiten' inputs (each feature separately) to mean 0
% if ~exist('meanAll','var')
%     allFeats = [];
%     for i = 1:length(allData)
%         allFeats = [allFeats ; allData{i}.feat2];
%     end
%     meanAll = mean(allFeats) + 1e-5;
%     stdAll  = std(allFeats) + 1e-5;
%     save('common/mean_std_all.mat', 'meanAll', 'stdAll');
%     clear allFeats;
% else
%     load('common/mean_std_all.mat' ,'meanAll','stdAll');
% end

% % ***TODO***: use generate_sds.m to whiten features
% % normalize features
% for i = 1:length(allData)
%     featsNow = allData{i}.feat2;
%     featsNow = bsxfun(@minus, featsNow, meanAll);
%     % Truncate to +/-3 standard deviations and scale to -1 to 1
%     pstd = 3 * stdAll;
%     featsNow = bsxfun(@max,bsxfun(@min,featsNow,pstd),-pstd);
%     featsNow = bsxfun(@times,featsNow,1./pstd);
% %     if strcmp(params.actFunc,'sigmoid')
% %         % Rescale from [-1,1] to [0.1,0.9]
% %         featsNow = (featsNow + 1) * 0.4 + 0.1;
% %     end
%     allData{i}.feat2 = featsNow;
%     clear featsNow;
% end

% assign each segment a label (by pixel majority vote from the annotated regions in labels)
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

tr_feat = [];
tr_label = [];
for i = 1:length(allData),
    if ~mod(i,10), fprintf('.'); end
    if ~mod(i,500), fprintf('%d\n',i); end
    
    % GT
    tr_label = [tr_label multi_output(allData{i}.segLabels,nlabel)];
    
    % read superpixel features
    feat = allData{i}.feat2;
    numFeat = size(feat, 1);
    feat = [feat, ones(numFeat, 1)];
    tr_feat = [tr_feat feat'];
    
    clear feat numFeat;
end

options.Method = 'lbfgs';
options.maxIter = 1500;
options.MaxFunEvals = 1500;
options.display = 'on';

num_in = size(tr_feat,1);
num_out = nlabel;
lrl2reg = 0.001;

W = 0.1*randn(num_in,num_out);
theta = W(:);

% train spatial logistic regression
[opttheta, ~] = minFunc( @(p) cost_lr(p,tr_feat,tr_label,num_in,num_out,lrl2reg), theta, options);

w_lr.nodeWeights = reshape(opttheta,num_in,num_out);
fname_lr = 'lr_without_whiten_feat';
save(sprintf('%s/%s.mat',fsave_dir,fname_lr),'w_lr');

% visualize weights, left 8 col for color, right 8 col for texture
for l = 1:nlabel
    figure, hintonDiagram(reshape(w_lr.nodeWeights(1:128,l), 8, 16));
end
