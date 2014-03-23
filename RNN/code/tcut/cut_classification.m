% test cut classification

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

%% get edge features, positive: cut, negative: non-cut

ndim = size(allTrees{i}.nodeFeatures,1)*3+1;
nedges = 30000;
nclass = 2;
edge_feats = zeros(ndim, nedges);
edge_labs = zeros(1, nedges);

cnt = 1;
for i = 1:length(allTrees)
  imgTree = allTrees{i};
  nLeafNodes = length(allData{i}.segLabels);
  minScore = min(imgTree.nodeScores);
  %for c = 1:nLeafNodes
  for c = 1:length(allcuts{i})
    par = imgTree.pp(c); 
    if par == 0
      continue
    end
    sibling = imgTree.kids(par,:);
    sibling = sibling(sibling ~= c);
    fprintf('tree %d - cut %d\n', i, c);
    edge_feats(:,cnt) = [imgTree.nodeFeatures(:,c); imgTree.nodeFeatures(:,sibling); ...
                        imgTree.nodeFeatures(:,par); ...
                        imgTree.nodeScores(par)-minScore];
    %edge_feats(:,cnt) = edge_feats(:,cnt) ./ norm(edge_feats(:,cnt));
    if allcuts{i}(c) 
      edge_labs(cnt) = 1;
    else
      edge_labs(cnt) = 0;
    end
    cnt = cnt + 1;
  end
end
temp_edge_feats = edge_feats(:, 1:cnt-1);
temp_edge_labs = edge_labs(:, 1:cnt-1);

% select some negative samples to make LR balance
stride = 5;
% neg_cand = edge_labs == 0;
% edge_feats = [temp_edge_feats(:,~neg_cand), temp_edge_feats(:,neg_cand)];
% edge_labs = [temp_edge_labs(:,~neg_cand), temp_edge_labs(:,neg_cand)];
neg_cand = find(temp_edge_labs == 0);
pos_cand = find(temp_edge_labs == 1);
neg_cand = neg_cand(1:stride:end);
edge_feats = [temp_edge_feats(:,pos_cand), temp_edge_feats(:,neg_cand)];
edge_labs = [temp_edge_labs(:,pos_cand), temp_edge_labs(:,neg_cand)];

%clear temp_edge_feats temp_edge_labs
%save edge_feats_labs.mat edge_feats edge_labs

train_feats = edge_feats(:, 1:2:end);
train_labs = edge_labs(:, 1:2:end);
test_feats = edge_feats(:, 2:2:end);
test_labs = edge_labs(:, 2:2:end);

%% train and test with logistic regression

net = glm(ndim, nclass-1, 'logistic');
options = foptions;
options(1) = 1;
options(14) = 50;
lr_net = glmtrain(net, options, train_feats', train_labs');

%save edge_lr_net.mat lr_net

pred = glmfwd(lr_net, test_feats');
pred_t = glmfwd(lr_net, train_feats');

pred_labs = pred >= 0.5;
pred_labs_t = pred_t >= 0.5;

test_err = sum(pred_labs' ~= test_labs) / numel(test_labs);
train_err = sum(pred_labs_t' ~= train_labs) / numel(train_labs);
fprintf('lr: test error = %f\n', test_err);
fprintf('lr: train error = %f\n', train_err);

%% MLP
% nhidden=20;
% nout=2;
% nin=100;
% alpha = 0.2;    % Weight decay
% ncycles = 200;   % Number of training cycles. 
% % Set up MLP network
% net = mlp(nin, nhidden, nout-1, 'logistic', alpha);
% options = zeros(1,18);
% options(1) = 1;                 % Print out error values
% options(14) = ncycles;
% net = netopt(net, options, train_feats', train_labs', 'quasinew');
% 
% save edge_mlp_net.mat net
% 
% pred = mlpfwd(net, test_feats');
% pred_t = mlpfwd(net, train_feats');
% 
% pred_labs = pred >= 0.5;
% pred_labs_t = pred_t >= 0.5;
% 
% test_err = sum(pred_labs' ~= test_labs) / numel(test_labs);
% train_err = sum(pred_labs_t' ~= train_labs) / numel(train_labs);
% fprintf('mlp: test error = %f\n', test_err);
% fprintf('mlp: train error = %f\n', train_err);

%% RBF SVM
% train_labs_svm = train_labs + 1;
% test_labs_svm = test_labs + 1;
% addpath ~/working/softwares/libsvm-3.17/matlab/
% svm_model = svmtrain(train_labs_svm', train_feats', '-t 2');
% [pred_labs, acc, scores] = svmpredict(test_labs_svm', test_feats', svm_model);

%% pegasos SVM
% vlabs = test_labs;
% vlabs(vlabs == 0) = -1;
% vlabs_t = train_labs;
% vlabs_t(vlabs_t == 0) = -1;
% lambda = 0.01;
% data = vl_maketrainingset(train_feats, int8(vlabs_t));
% [w b info] = vl_svmpegasos(data,lambda,'MaxIterations',5000);
% scores = w'*test_feats + b;
% figure; vl_pr(vlabs, scores);

%% evaluation
vlabs = test_labs;
vlabs(vlabs == 0) = -1;
figure; vl_pr(vlabs, pred);
figure; vl_roc(vlabs, pred);