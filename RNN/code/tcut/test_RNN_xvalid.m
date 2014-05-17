function [accs mprs] = test_RNN_xvalid()
% logistic regression with 5-fold cross validation
addpath ~/working/deep/netlab3_3/
addpath ../
addpath(genpath('../tools/'));

%%%%%%%%%%%%%%%%%%%%%%%
% minfunc options (not tuned)
options.Method = 'lbfgs';
options.MaxIter = 1000;
optionsPT=options;
options.TolX = 1e-4;


%%%%%%%%%%%%%%%%%%%%%%%
%iccv09: 0 void   1,1 sky  0,2 tree   2,3 road  1,4 grass  1,5 water  1,6 building  2,7 mountain 2,8 foreground
set(0,'RecursionLimit',1000);
params.numLabels = 7; % we never predict 0 (void)
params.numFeat = 268;


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

%%%%%%%%%%%%%%%%%%%%%%
% input and output file names
mainDataSet = 'iccv09';
dataSet = 'train';
neighNameStem = ['../../data/' mainDataSet '-allNeighborPairs'];
%neighName = [neighNameStem '_' dataSet '.mat'];

accs = zeros(1,5);
mprs = zeros(1,5);

%% prepare data
load ../../data/iccv09-allData-train.mat
[nr, nim] = size(allData);  % nr is 1, nim is the number of images
D = size(allData{1}.feat2,2); % number of features
nclass = 8;

if ~isfield(allData{1}, 'labelCountsPerSP')
  % numLabelsPerSP
  for i = 1:length(allData)
      labelRegs = allData{i}.labels;
      segs = allData{i}.segs2;
      numSegs = max(segs(:));
      labelCountsPerSP = zeros(numSegs, nclass);
      numPixelInSP = zeros(numSegs, 1);
      for r = 1:numSegs
          numPixelInSP(r) = numel(labelRegs(segs == r));
          for ci = 1:nclass
              labelCountsPerSP(r,ci) = sum(labelRegs(segs==r) == ci);
          end
      end
      allData{i}.labelCountsPerSP = labelCountsPerSP;
      allData{i}.numPixelInSP = numPixelInSP;
  end

  % expectation observation
  for i = 1:length(allData)
      labelRegs = allData{i}.labels;
      segs = allData{i}.segs2;
      numSegs = max(segs(:));
      expectSegLabels = zeros(numSegs, nclass);
      for r = 1:numSegs
          numPixelInSeg = numel(labelRegs(segs == r));
          for ci = 1:nclass
              expectSegLabels(r,ci) = sum(labelRegs(segs==r) == ci) / numPixelInSeg;
          end
      end
      allData{i}.expectSegLabels = expectSegLabels;
  end

  save ../../data/iccv09-allData-train.mat allData
end

allDataFold = cell(1,5);
for xv = 1:4
	allDataFold{xv} = allData(1,(xv-1)*143+1:xv*143);
end

nsp_fold = [];
nsp = 0; % number of superpixels
cnt = 0;
for i = 1:nim
  nsp = nsp +  size(allData{i}.segLabels,1);
  cnt = cnt + 1;
  if cnt >= 143
	  nsp_fold(end+1) = nsp;
	  cnt = 0;
	  nsp = 0;
  end
end
clear allData 

load ../../data/iccv09-allData-eval.mat
allDataFold{5} = allData;

nsp = 0;
for i = 1:143
	nsp = nsp + size(allData{i}.segLabels,1);
end
nsp_fold(end+1) = nsp;
clear allData 

%% training & testing
%load ../../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
%nets = cell(1,5);
for xv = 1:5
	fprintf('training in fold %d...\n', xv);
	%[Xe te] = form_data(allDataFold{xv}, nsp_fold{xv}, D);
  allData = [allDataFold{setdiff(1:5,xv)}];
	%[Xr tr] = form_data(allData, sum(nsp_fold)-nsp_fold(xv), D, nclass);
  
  neighName = [neighNameStem '_' dataSet 'fold' num2str(xv) '.mat'];
  if ~exist(neighName, 'file')
    preProSegFeatsAndSave2(allData, neighName, params);
  else
    load(neighName,'goodPairsL','goodPairsR','badPairsL','badPairsR','onlyGoodL','onlyGoodR','onlyGoodLabels','allSegs','allSegLabels');
  end
  
  paramString = ['_hid' num2str(params.numHid ) '_PTC' num2str(params.regPTC)];
  fullParamNameBegin = ['../../output/' mainDataSet '_fullParams'];
  paramString = [paramString '_fullC' num2str(params.regC) '_L' num2str(params.LossPerError)];
  fullTrainParamName = [fullParamNameBegin paramString '_fold' num2str(xv) '.mat'];
  disp(['fullTrainParamName=' fullTrainParamName ])
  
  %%%%%%%%%%%%%%%%%%%%%%
  % initialize parameters
  initParams

  %%%%%%%%%%%%%%%%%%%%%%
  % TRAINING

  % train Wbot layer and first RNN collapsing decisions with all possible correct and incorrect segment pairs
  % this uses the training data more efficiently than the purely greedy full parser training that only looks at some pairs
  % both could have been combined into one training as well.
  [X decodeInfo] = param2stack(Wbot,W,Wout,Wcat);
  X = minFunc(@costFctInitWithCat,X,optionsPT,decodeInfo,goodPairsL,goodPairsR,badPairsL,badPairsR,onlyGoodL,onlyGoodR,onlyGoodLabels,allSegs,allSegLabels,params);

  X = minFunc(@costFctFull,X,options,decodeInfo,allData,params);
  [Wbot,W,Wout,Wcat] = stack2param(X, decodeInfo);
  save(fullTrainParamName,'Wbot','W','Wout','Wcat','params','options');

	% compute all parse trees
  clear allData
  allData = allDataFold{xv};
  tree_file = ['../../output/iccv09-new-allTrees-fold' num2str(xv) '.mat'];
  if ~exist('allTrees','var')
      if exist(tree_file,'file')
          load(tree_file); 
      else
          allTrees = cell(1,length(allData));
          for i = 1:length(allData)
              if length(allData{i}.segLabels)~=size(allData{i}.feat2,1)
                  disp(['Image ' num2str(i) ' has faulty data, skipping!'])
                  continue
              end
              topCorr=0;
              imgTreeTop = parseImage(topCorr,Wbot,W,Wout,Wcat,allData{i}.adj, ...
                  allData{i}.feat2,allData{i}.segLabels,params);
              allTrees{i} = imgTreeTop;
          end
          save(tree_file, 'allTrees');
      end
  end
  
%   p_connect_star = zeros(1,length(allData));
%   for i = 1:length(allData)
% 
%       % get which p_connect will be used
%       [~,~,~,~,q_max_diff,q_max_diff_ind] = tree_cut_new(allData{i}, allTrees{i}, theta_plus, nclass, -1);
% 
%       p_samples = unique(q_max_diff);
%       p_samples = 1./(1+8./exp(p_samples)); % 1 / ( 1 + ( n/exp(q_diff) ) )
%       assert(all(p_samples >= 0 & p_samples <= 1));
%       p_samples = sort(p_samples(p_samples > 0));
%       p_samples = [linspace(0,p_samples(1),10), linspace(p_samples(1),1,5), p_samples, 1/9];
%       p_samples = unique(p_samples);
%       p_samples = sort(p_samples);
% 
%       [PRs,spAccs,nCuts,PRs3,GCEs,VIs] = evalSegPerImg([], @tree_cut_new, allData{i}, allTrees{i}, theta_plus, nclass, p_samples,0,0);
%       combo = (PRs.*spAccs.*PRs3) ./ (nCuts.*(VIs+1e-5));
%       %[peak,loc] = max(PRs);
%       [peak,loc] = fullmax(combo);
%       [mnc,mmi] = min(nCuts(loc));
%       p_connect_star(i) = p_samples(loc(mmi));
% 
%       %fprintf('>>>>>> image %d: best combo value = %f at %f\n', i, peak(mmi), p_samples(loc(mmi)));
%       %fprintf('------ forest: number of subtrees = %d\n', mnc);
%   end

  Z = cell(1, length(allData));
  for j = 1:length(allData)
    Z{j} = allTrees{j}.catOut;
  end
  
	[accs(xv) mprs(xv)] = evaluateImgPixels2(allData, Z);
  %[accs(xv) mprs(xv)] = evaluateImgPixels3(allData, allTrees, allppred);
	fprintf('fold %d: acc = %f, mpr = %\n', xv, accs(xv), mprs(xv));
  clear allTrees
end


%% helper functions
function [X t y] = form_data(allData, nsp, D, nclass)

nim = length(allData);
X = zeros(nsp,D);  % data matrix
y = zeros(nsp,1);  % holds labels

count = 1;  % counter for each row of X
for i = 1:nim
     spi = size(allData{i}.feat2,1);  % number of SP in this image
     X(count:count+spi-1,:) =  allData{i}.feat2;
     y(count:count+spi-1,:) =  allData{i}.segLabels;
     count = count + spi;
end
assert(max(y) == nclass);

% remove y = 0
indx = y ~= 0;
y = y(indx);
X = X(indx,:);

% recode y in one-of-nclass format
id = eye(nclass);
t = id(y,:);

function preProSegFeatsAndSave2(allData, neighName, params, mainDataSet)

if nargin < 4
  mainDataSet = 'iccv09';
end

% collect all good and bad segment pairs
% pre-allocate (and later delete empty rows)
if strcmp(mainDataSet,'msrc')
    upperBoundSegPairsNum = length(allData) * 600*10;
else
    upperBoundSegPairsNum = length(allData) * 150*5;
end
goodPairsL = zeros(params.numFeat+1,upperBoundSegPairsNum);
goodPairsR = zeros(params.numFeat+1,upperBoundSegPairsNum);
badPairsL = zeros(params.numFeat+1,upperBoundSegPairsNum);
badPairsR = zeros(params.numFeat+1,upperBoundSegPairsNum);
startBoth = 1;
startBad = 1;

onlyGoodL = zeros(params.numFeat+1,upperBoundSegPairsNum);
onlyGoodR = zeros(params.numFeat+1,upperBoundSegPairsNum);
onlyGoodLabels = zeros(1,upperBoundSegPairsNum);
startOnlyGood = 1;

allSegs = zeros(params.numFeat+1,upperBoundSegPairsNum);
allSegLabels =  zeros(1,upperBoundSegPairsNum);
startAllSegs = 1;

for i = 1:length(allData)
    segs = allData{i}.segs2;
    feats = allData{i}.feat2;
    segLabels = allData{i}.segLabels;
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % find neighbors!
    adjHigher = getAdjacentSegments(segs);%getAdjacentSegments(segs,1)
    adj = adjHigher|adjHigher';
    allData{i}.adj = adj;
    % compute only all pairs and train to merge or not
    for s = 1:length(segLabels)
        % save all segs and their labels for pre-training
        if segLabels(s)>0
            allSegs(:,startAllSegs)= [feats(s,:)' ;1];
            allSegLabels(startAllSegs) = segLabels(s);
            startAllSegs=startAllSegs+1;
        end
        
        neighbors = find(adj(s,:));
        sameLabelNeigh = segLabels(neighbors)==segLabels(s);
        goodNeighbors = neighbors(sameLabelNeigh);
        badNeighbors = neighbors(~sameLabelNeigh);
        numGood = length(goodNeighbors);
        numBad = length(badNeighbors);
        numGBPairs = numGood * numBad;
        
        % never train on void segments: !
        if segLabels(s)>0
            for g = 1:numGood
                onlyGoodL(:,startOnlyGood:startOnlyGood+numGood-1)= [repmat(feats(s,:)',1,numGood ) ;ones(1,numGood)];
                onlyGoodR(:,startOnlyGood:startOnlyGood+numGood-1)= [feats(goodNeighbors,:)' ;ones(1,numGood)];
                onlyGoodLabels(startOnlyGood:startOnlyGood+numGood-1) = segLabels(s);
            end
            startOnlyGood = startOnlyGood + numGood;
        end
        
        if numGood>0 && numBad>0
            gbPairNums = cartprod(goodNeighbors,badNeighbors);
            % these are the inputs to Wbot
            goodPairsL(:,startBoth:startBoth+numGBPairs-1)= [repmat(feats(s,:)',1,numGBPairs) ;ones(1,numGBPairs)];
            goodPairsR(:,startBoth:startBoth+numGBPairs-1)= [feats(gbPairNums(:,1),:)' ;ones(1,numGBPairs)];
            
            badPairsL(:,startBoth:startBoth+numGBPairs-1)= [repmat(feats(s,:)',1,numGBPairs) ;ones(1,numGBPairs)];
            badPairsR(:,startBoth:startBoth+numGBPairs-1)= [feats(gbPairNums(:,2),:)' ;ones(1,numGBPairs)];
            
            startBoth = startBoth+numGBPairs;
        end
        
    end
    if mod(i,20)==0, disp([num2str(i) '/' num2str(length(allData))]);end
end

numAllSegs = startAllSegs-1;
allSegs= allSegs(:,1:numAllSegs);
allSegLabels= allSegLabels(1:numAllSegs);

numOnlyGood = startOnlyGood-1;
onlyGoodL = onlyGoodL(:,1:numOnlyGood);
onlyGoodR = onlyGoodR(:,1:numOnlyGood);
onlyGoodLabels= onlyGoodLabels(1:numOnlyGood);

numGBPairsAll = startBoth-1;
% delete trailing zeros in pre-allocated matrix
goodPairsL = goodPairsL(:,1:numGBPairsAll);
goodPairsR = goodPairsR(:,1:numGBPairsAll);
badPairsL = badPairsL(:,1:numGBPairsAll);
badPairsR = badPairsR(:,1:numGBPairsAll);

save(neighName,'goodPairsL','goodPairsR','badPairsL','badPairsR','meanAll','stdAll','onlyGoodL','onlyGoodR','onlyGoodLabels','allSegs','allSegLabels');
