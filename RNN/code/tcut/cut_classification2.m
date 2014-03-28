% test cut classification
% train and test use different subsets of trees.

%% setting and loading data
addpath ../
load ../../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
load ../../data/iccv09-allData-eval.mat
run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');
addpath ~/working/deep/netlab3_3/
theta_plus = MLE_theta(allData,8);

tree_file = '../../output/iccv09-allTrees-eval.mat';
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
        save('../../output/iccv09-allTrees-eval.mat', 'allTrees');
    end
end

% to get p_connect_star
if ~exist('p_connect_star', 'var')
  if exist('p_connect_star.mat', 'file')
    load p_connect_star.mat
  else
    test_PR_new;
  end
end

%% compute cuts using p_connect_star
allQ = cell(1,length(allData));
allcuts = cell(1,length(allData));
alllabs = cell(1,length(allData));
allforest = cell(1,length(allData));

for i = 1:length(allData)
  [allQ{i},allcuts{i},alllabs{i},allforest{i}] = tree_cut_new(allData{i}, allTrees{i}, theta_plus, 8, p_connect_star(i));
end

%% marginal probs
allMarginals = cell(1,length(allData));
for i = 1:length(allData)
  allMarginals{i} = allQ{i}';
end

%% structural features
if exist('CCsUnder.mat', 'file')
  load CCsUnder.mat
else
  allnumLeafsUnder = cell(1,length(allData));
  allleafsUnder = cell(1,length(allData));
  allCCsUnder = cell(1,length(allData));
  nlabs = 8;
  for i = 1:length(allData)
    % leafs of each subtree
    numLeafNodes = size(allData{i}.adj,1);
    numTotalNodes = size(allTrees{i}.kids,1);
    numLeafsUnder = ones(numLeafNodes,1);
    leafsUnder = cell(numLeafNodes,1);
    CCsUnder = ones(numTotalNodes,1);

    for s = 1:numLeafNodes
        leafsUnder{s} = s;
    end

    spLabelGT = zeros(size(allData{i}.segs2));
    % get SP GT
    for j = 1:length(allData{i}.segLabels)
        spLabelGT(allData{i}.segs2 == j) = allData{i}.segLabels(j);
    end

    for n = numLeafNodes+1:numTotalNodes
        kids = allTrees{i}.getKids(n);
        numLeafsUnder(n) = numLeafsUnder(kids(1))+numLeafsUnder(kids(2));
        leafsUnder{n} = [leafsUnder{kids(1)} leafsUnder{kids(2)}];

        subSPGT = zeros(size(spLabelGT));
        subIndex = logical(sum(bsxfun(@eq, allData{i}.segs2(:), leafsUnder{n}),2));
        subSPGT(subIndex) = spLabelGT(subIndex);

        for k = 1:nlabs
          temp = bwlabel(subSPGT == k);
          CCsUnder(n) = CCsUnder(n) + numel(unique(temp(temp > 0)));
        end
    end

    allnumLeafsUnder{i} = numLeafsUnder;
    allleafsUnder{i} = leafsUnder;
    allCCsUnder{i} = CCsUnder;
  end
end

%% get edge features, positive: cut, negative: non-cut
train_index = 1:2:length(allTrees);
test_index = 2:2:length(allTrees);

ndim = 7;
% ndim = size(allTrees{i}.nodeFeatures,1)*3+size(allTrees{i}.catOut,1)*3+7;
nedges = 30000;
nclass = 2;

% -- training data
table_diff_cc_train = zeros(2,200);

edge_feats = zeros(ndim, nedges);
edge_labs = zeros(1, nedges);

cnt = 1;
for i = train_index
  imgTree = allTrees{i};
  nLeafNodes = length(allData{i}.segLabels);
  minScore = min(imgTree.nodeScores);
  
  for c = 1:length(allcuts{i})
    par = imgTree.pp(c); 
    if par == 0
      continue
    end
    sibling = imgTree.kids(par,:);
    sibling = sibling(sibling ~= c);
    fprintf('tree %d - cut %d\n', i, c);

    edge_feats(:,cnt) = [imgTree.nodeScores(par); ...
                        allCCsUnder{i}(c);allCCsUnder{i}(sibling);allCCsUnder{i}(par); ...
                        allnumLeafsUnder{i}(c);allnumLeafsUnder{i}(sibling);allnumLeafsUnder{i}(par)
                        ];

    % normalization
    edge_feats(:,cnt) = edge_feats(:,cnt) ./ norm(edge_feats(:,cnt));                  
    if allcuts{i}(c) 
      edge_labs(cnt) = 1;
    else
      edge_labs(cnt) = 0;
    end
    
    diff_cc = allCCsUnder{i}(par) - allCCsUnder{i}(c) + 100;
    table_diff_cc_train(edge_labs(cnt)+1, diff_cc) = table_diff_cc_train(edge_labs(cnt)+1, diff_cc) + 1;
    
    cnt = cnt + 1;
  end
end
temp_edge_feats = edge_feats(:, 1:cnt-1);
temp_edge_labs = edge_labs(:, 1:cnt-1);

% select some negative samples to make LR balance
stride = 1;
neg_cand = find(temp_edge_labs == 0);
pos_cand = find(temp_edge_labs == 1);
neg_cand = neg_cand(1:stride:end);
edge_feats = [temp_edge_feats(:,pos_cand), temp_edge_feats(:,neg_cand)];
edge_labs = [temp_edge_labs(:,pos_cand), temp_edge_labs(:,neg_cand)];

%clear temp_edge_feats temp_edge_labs
%save edge_feats_labs.mat edge_feats edge_labs

train_feats = edge_feats;
train_labs = edge_labs;

%-- testing data
table_diff_cc_test = zeros(2,200);

edge_feats = zeros(ndim, nedges);
edge_labs = zeros(1, nedges);

cnt = 1;
for i = test_index
  imgTree = allTrees{i};
  nLeafNodes = length(allData{i}.segLabels);
  minScore = min(imgTree.nodeScores);
  
  for c = 1:length(allcuts{i})
    par = imgTree.pp(c); 
    if par == 0
      continue
    end
    sibling = imgTree.kids(par,:);
    sibling = sibling(sibling ~= c);
    fprintf('tree %d - cut %d\n', i, c);

    edge_feats(:,cnt) = [imgTree.nodeScores(par); ...
                        allCCsUnder{i}(c);allCCsUnder{i}(sibling);allCCsUnder{i}(par); ...
                        allnumLeafsUnder{i}(c);allnumLeafsUnder{i}(sibling);allnumLeafsUnder{i}(par)
                        ];

    % normalization
    edge_feats(:,cnt) = edge_feats(:,cnt) ./ norm(edge_feats(:,cnt));                  
    if allcuts{i}(c) 
      edge_labs(cnt) = 1;
    else
      edge_labs(cnt) = 0;
    end
    
    diff_cc = allCCsUnder{i}(par) - allCCsUnder{i}(c) + 100;
    table_diff_cc_test(edge_labs(cnt)+1, diff_cc) = table_diff_cc_test(edge_labs(cnt)+1, diff_cc) + 1;
    
    cnt = cnt + 1;
  end
end
temp_edge_feats = edge_feats(:, 1:cnt-1);
temp_edge_labs = edge_labs(:, 1:cnt-1);

% select some negative samples to make LR balance
stride = 1;
neg_cand = find(temp_edge_labs == 0);
pos_cand = find(temp_edge_labs == 1);
neg_cand = neg_cand(1:stride:end);
edge_feats = [temp_edge_feats(:,pos_cand), temp_edge_feats(:,neg_cand)];
edge_labs = [temp_edge_labs(:,pos_cand), temp_edge_labs(:,neg_cand)];

test_feats = edge_feats;
test_labs = edge_labs;

%% train and test with logistic regression

% training
net = glm(ndim, nclass-1, 'logistic');
options = foptions;
options(1) = 1;
options(14) = 50;
lr_net = glmtrain(net, options, train_feats', train_labs');

%save edge_lr_net.mat lr_net

% testing
pred = glmfwd(lr_net, test_feats');
pred_t = glmfwd(lr_net, train_feats');

pred_labs = pred >= 0.5;
pred_labs_t = pred_t >= 0.5;

test_err = sum(pred_labs' ~= test_labs) / numel(test_labs);
train_err = sum(pred_labs_t' ~= train_labs) / numel(train_labs);
fprintf('lr: test error = %f\n', test_err);
fprintf('lr: train error = %f\n', train_err);

%% evaluation
% vlabs = test_labs;
% vlabs(vlabs == 0) = -1;
% figure; vl_pr(vlabs, pred);
% figure; vl_roc(vlabs, pred, 'plot', 'fptp');

%% p_connect_pred
allppred = cell(1,length(allData));

for i = 1:length(allData)
  imgTree = allTrees{i};
  nLeafNodes = length(allData{i}.segLabels);
  minScore = min(imgTree.nodeScores);
  
  edge_feats = zeros(ndim, length(allcuts{i}));
  edge_labs = zeros(1, length(allcuts{i}));
  
  for c = 1:length(allcuts{i})
    par = imgTree.pp(c); 
    if par == 0
      continue
    end
    sibling = imgTree.kids(par,:);
    sibling = sibling(sibling ~= c);
    fprintf('tree %d - cut %d\n', i, c);

    edge_feats(:,c) = [imgTree.nodeScores(par); ...
                        allCCsUnder{i}(c);allCCsUnder{i}(sibling);allCCsUnder{i}(par); ...
                        allnumLeafsUnder{i}(c);allnumLeafsUnder{i}(sibling);allnumLeafsUnder{i}(par)
                        ];

    % normalization
    edge_feats(:,c) = edge_feats(:,c) ./ norm(edge_feats(:,c));  
    
    if allcuts{i}(c) 
      edge_labs(c) = 1;
    else
      edge_labs(c) = 0;
    end 
    
  end
  
  pred = glmfwd(lr_net, edge_feats');
  allppred{i} = 1 - pred;
end

%% new cuts
allQ_pred = cell(1,length(allData));
allcuts_pred = cell(1,length(allData));
alllabs_pred = cell(1,length(allData));
allforest_pred = cell(1,length(allData));

for i = 1:length(allData)
  [allQ_pred{i},allcuts_pred{i},alllabs_pred{i},allforest_pred{i}] = ...
    tree_cut_new(allData{i}, allTrees{i}, theta_plus, 8, allppred{i});
end

%% statistics of dataset
totNumCuts = zeros(1,length(allData));
totNumCuts_pred = zeros(1,length(allData));
totNumCCs = zeros(1,length(allData));
for i = 1:length(allData)
  totNumCuts(i) = sum(allcuts{i});
  totNumCuts_pred(i) = sum(allcuts_pred{i});
  totNumCCs(i) = allCCsUnder{i}(end);
end

maxCCs = max(totNumCCs);
maxCuts = max(totNumCuts);
maxCuts_pred = max(totNumCuts_pred);

tot_index_same = totNumCuts == totNumCuts_pred;

figure; scatter(totNumCCs, totNumCuts, 'filled'); xlabel('Number of CCs'); ylabel('Number of Cuts');
hold on; line(1:max(maxCCs,maxCuts), 1:max(maxCCs,maxCuts), 'Color','g', 'LineStyle', '--', 'LineWidth', 2);
figure; scatter(totNumCCs(tot_index_same), totNumCuts_pred(tot_index_same), 'filled', 'b'); xlabel('Number of CCs'); ylabel('Number of Cuts');
hold on; scatter(totNumCCs(~tot_index_same), totNumCuts_pred(~tot_index_same), 'filled', 'r'); xlabel('Number of CCs'); ylabel('Number of Cuts');
hold on; line(1:max(maxCCs,maxCuts_pred), 1:max(maxCCs,maxCuts_pred), 'Color','g', 'LineStyle', '--', 'LineWidth', 2);

%% PR value comparison
vis = 0;
vtree = 0;

PR_peaks = zeros(1,length(allData));
SPAccsMax = zeros(1,length(allData));
%p_connect_star = zeros(1,length(allData));
for i = 1:length(allData)
    if vis && vtree
      name = ['iccv09/iccv09_' num2str(i) '_'];
    else
      name = [];
    end
    
    [PRs,spAccs,nCuts,PRs3,GCEs,VIs] = evalSegPerImg3(name, @tree_cut_new, allData{i}, allTrees{i}, theta_plus, 8, allppred{i},vis,vtree);

    PR_peaks(i) = PRs;
    SPAccsMax(i) = max(spAccs);

    fprintf('>>>>>> image %d: PR value = %f, SPAcc = %f', i, PRs, spAccs);
    fprintf('------ forest: number of subtrees = %d\n', nCuts);
    if vis && isempty(name)
        pause
    end
end

PR_peaks_pred = zeros(1,length(allData));
SPAccsMax_pred = zeros(1,length(allData));
%p_connect_star = zeros(1,length(allData));
for i = 1:length(allData)
    if vis && vtree
      name = ['iccv09/iccv09_' num2str(i) '_'];
    else
      name = [];
    end
    
    [PRs,spAccs,nCuts,PRs3,GCEs,VIs] = evalSegPerImg3(name, @tree_cut_new, allData{i}, allTrees{i}, theta_plus, 8, p_connect_star(i),vis,vtree);
    
    PR_peaks_pred(i) = PRs;
    SPAccsMax_pred(i) = max(spAccs);

    fprintf('>>>>>> image %d: PR value = %f, SPAcc = %f', i, PRs, spAccs);
    fprintf('------ forest: number of subtrees = %d\n', nCuts);
    if vis && isempty(name)
        pause
    end
end

min_PR = min([PR_peaks, PR_peaks_pred]);
max_PR = max([PR_peaks, PR_peaks_pred]);
min_Acc = min([SPAccsMax, SPAccsMax_pred]);
max_Acc = max([SPAccsMax, SPAccsMax_pred]);

figure; scatter(PR_peaks, PR_peaks_pred, 'filled'); xlabel('PRs Global'); ylabel('PRs local');
hold on; line([min_PR max_PR], [min_PR max_PR], 'Color','g', 'LineStyle', '--', 'LineWidth', 2);
figure; scatter(SPAccsMax, SPAccsMax_pred, 'filled', 'b'); xlabel('SP Accuracy Global'); ylabel('SP Accuracy local');
hold on; line([min_Acc max_Acc], [min_Acc max_Acc], 'Color','g', 'LineStyle', '--', 'LineWidth', 2);