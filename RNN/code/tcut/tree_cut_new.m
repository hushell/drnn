function [Q,cuts,labs,forest, q_max_diff, q_max_diff_ind] = tree_cut_new(...
  imgData, imgTreeTop, theta_plus, n_labs, p_connect, vis, debug)
% DP: tree-cut algorithm
% Q(t,z) = max [ Q(l,z) + Q(r,z) + prior_merge, max_z Q(l,z) + Q(r,z) + prior_cut, 
%                Q(l,z) + max_z Q(r,z) + prior_cut]
%
% @param imgData: image related data
% @param imgTreeTop: RNN tree
% @param theta_plus: MLE estimated parameters
% @param n_labs: number of classes
% @param p_connect: prior of child-parent connection
% @param vis: whether visualize the segmentation
% @param debug: use subtree for debugging
% NOTE: for visualizing, you need to have vlfeat somewhere and setup it, e.g.
% run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');
%
% @return Q: maximum likelihood for each node
% @return cuts: a indicator vector, cuts(j) = 1 means there is a cut above node j
% @return labs: predicted labels of leafs
% @return forest: forest(j) = i, means that leaf j is under subtree rooted at i
%
% example: [Q,cuts,labels,forest] = tree_cut_new(allData{4}, allTrees{4}, theta_plus, 8, 0.13, 0, 0);


if nargin < 6
    vis = 0;
    debug = 0;
end

if nargin < 7
    debug = 0;
end

%% global variables
global q cut_at cut_if pred_labs

numLeafNodes = size(imgData.adj,1);
numTotalNodes = size(imgTreeTop.kids,1);

if debug
    numTotalNodes = 173; % for allTrees{1}, it's a subtree with 4 leafs
    %numTotalNodes = 224;
end

%% leafs of each subtree
numLeafsUnder = ones(numLeafNodes,1);
leafsUnder = cell(numLeafNodes,1);

for s = 1:numLeafNodes
    leafsUnder{s} = s;
end

for n = numLeafNodes+1:numTotalNodes
    kids = imgTreeTop.getKids(n);
    numLeafsUnder(n) = numLeafsUnder(kids(1))+numLeafsUnder(kids(2));
    leafsUnder{n} = [leafsUnder{kids(1)} leafsUnder{kids(2)}];
end

%% counts of pixel for each subtree
% USE counts (GT)
csp = imgData.labelCountsPerSP; 
% USE conditional likelihood
%csp = imgTreeTop.catOut(:,1:numLeafNodes)' .* repmat(imgData.numPixelInSP, 1, n_labs);
%csp = imgTreeTop.catOut(:,1:numLeafNodes)';
count_csp = sum(csp,2);

% compute likelihood for each superpixel
% USE P(Y_j | z_j) = nml * Y_jz log(theta_plus / theta_minus) + \sum_{k} Y_jk log(theta_minus)
theta_minus = (1 - theta_plus) ./ (n_labs - 1);
l_theta_plus = log(theta_plus);
l_theta_minus = log(theta_minus);
l_theta_diff = l_theta_plus - l_theta_minus;

% factorial(n) = gamma(n+1), nml = n!/(x_1!...x_n!)
nml = gammaln(sum(csp,2)+1)-sum(gammaln(csp+1),2);
lik = csp .* repmat(l_theta_diff, numLeafNodes, 1) ... % Y_jz log(theta_plus / theta_minus
    + repmat(count_csp,1,n_labs) .* repmat(l_theta_minus, numLeafNodes, 1) ... % \sum_{k} Y_jk log(theta_minus)
    + repmat(nml,1,n_labs); % log(nml)

%% bottom-up phase: Q(t) 
q = zeros(numTotalNodes, n_labs); % max likelihood
q(1:numLeafNodes,:) = lik;

% track decisions, values range from 1: merge; 2: cutL; 3: cutR
cut_at = zeros(numTotalNodes,n_labs); 

% compute prior_merge & prior_cut
if p_connect < 0
    prior_merge = 0;
    prior_cut = 0;
elseif p_connect == 0
    prior_merge = -Inf;
    prior_cut = 0;
else
    const1 = 1;
    prior_merge = log(p_connect) + log(p_connect); prior_merge = prior_merge * const1;
    prior_cut = log(p_connect) + log(1-p_connect) - log(n_labs); prior_cut = prior_cut * const1;
    %prior_cut = log(p_connect) + log(1-p_connect); prior_cut = prior_cut * const1 - log(n_labs);
end
fprintf('p: %f, prior_merge = %f, prior_cut = %f, diff = %f\n', ...
    p_connect, prior_merge, prior_cut, prior_merge-prior_cut);
  
% for finding PR
%q_max_diff = zeros(1,numTotalNodes-numLeafNodes);
%q_max_diff_ind = numLeafNodes+1:numTotalNodes;
q_max_diff = [];
q_max_diff_ind = [];

% DP loop
for j = numLeafNodes+1:numTotalNodes
    kids = imgTreeTop.getKids(j);
    L = kids(1); R = kids(2);
    
    [maxL,posL] = max(q(L,:));
    [maxR,posR] = max(q(R,:));
    %maxL = 0;
    %maxR = 0;
    
    %prior_merge = 0.1;
    %prior_cut = 0;
    
    q_merge = q(L,:) + q(R,:) + prior_merge;
    q_cutL = maxL + q(R,:) + prior_cut;
    q_cutR = maxR + q(L,:) + prior_cut;
    
    q_all = [q_merge; q_cutL; q_cutR];
    [q(j,:), cut_at(j,:)] = max(q_all); % for each state, see which decision maximize Q
    
    if vis
      fprintf('------------ node %d -------------\n', j);

      fprintf('q_L %d = \n', L); disp(q(L,:));
      fprintf('q_R %d = \n', R); disp(q(R,:));

      fprintf('q_j %d = \n', j); disp(q(j,:));
      fprintf('cut_at_where %d = \n', j); disp(cut_at(j,:));
    end
    
    %q_max_diff(j-numLeafNodes) = max(max(q_all(2:end,:))) - max(q_all(1,:));
    q_diff = max(q_all(2:end,:)) - q_all(1,:);
    q_max_diff = [q_max_diff q_diff(q_diff > 0)];
end

if p_connect < 0
    [q_max_diff, tmp_ind] = sort(q_max_diff);
    %q_max_diff_ind = q_max_diff_ind(tmp_ind);
end

%% top down decoding
cut_if = zeros(numTotalNodes,1); % if cut above
cut_if(end) = 1; % always cut above root node

pred_labs = zeros(numTotalNodes,1); % predicted label for each node
j = numTotalNodes;
[~,state] = max(q(j,:));
pred_labs(j) = state;

top_down_decoding(imgTreeTop, j, state); % output cut_if

% get forest
forest = zeros(numTotalNodes,1); % indicate leafs belong to which tree 

% find the lowest cuts
for i = 1:numTotalNodes
    j = i;
    while 0 ~= j % parent(top) = 0
        if cut_if(j) == 1 
            forest(i) = j;
            break
        end
        j = imgTreeTop.pp(j); % go to parent
    end
end

%% final outputs and visualization
labs = pred_labs;
cuts = cut_if;
%Q = max(q,[],2);
qind = sub2ind(size(q),1:numel(pred_labs),pred_labs');
Q = q(qind);

if vis
    figure(p_connect*100000+1);
    %figure(100);
    colorImgWithLabels_vlfeat(imgData.segs2,imgData.labels,pred_labs,...
            imgData.segLabels, imgData.img);

    sp_map = zeros(size(imgData.labels));
    for i = 1:length(forest)
        sp_map(imgData.segs2 == i) = forest(i);
    end

    figure(p_connect*100000+2);
    %figure(101);
    imagesc(sp_map);
    title(sprintf('There are %d segments (subtrees) in the forest', sum(cut_if)));
end

%% helper functions
function top_down_decoding(imgTreeTop, j, state)
% pre-order traversal
global cut_at q cut_if pred_labs

kids = imgTreeTop.getKids(j);
L = kids(1);
R = kids(2);
if L == 0
    return
end

decis = cut_at(j,state); % decision for this state

if decis == 1 % merge
    stateL = state;
    stateR = state;
elseif decis == 2 % cutL
    cut_if(kids(1)) = 1;
    [~,stateL] = max(q(L,:));
    stateR = state;
elseif decis == 3 % cutR
    cut_if(kids(2)) = 1;
    [~,stateR] = max(q(R,:));
    stateL = state;
else
    disp('*** Something Wrong!');
    return
end

pred_labs(L) = stateL;
pred_labs(R) = stateR;

top_down_decoding(imgTreeTop, L, stateL);
top_down_decoding(imgTreeTop, R, stateR);


