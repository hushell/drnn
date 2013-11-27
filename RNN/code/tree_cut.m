function [Q,cuts,labs,forest] = tree_cut(imgData, imgTreeTop, theta_plus, n_labs, p_connect, vis, debug)
% DP: tree-cut algorithm
% Q(t,z) = max [ Q(l,z) + Q(r,z), max_z Q(l,z) + Q(r,z) - lambda, 
%                Q(l,z) + max_z Q(r,z) - lambda ]
%
% @param imgData: image related data
% @param imgTreeTop: RNN tree
% @param theta_plus: MLE estimated parameters
% @param n_labs: number of classes
% @param p_connect: penalty for introducing a cut
% @param vis: whether visualize the segmentation
% @param debug: use subtree for debugging
% NOTE: for visualizing, you need to have vlfeat somewhere and setup it, e.g.
% run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');
%
% @return Q: maximum likelihood for each node
% @return cuts: a indicator vector, cuts(j) = 1 means there is a cut above node j
% @return labs: predicted labels of leafs
%
% example: [Q,cuts,labels] = tree_cut(allData{i}, allTrees{i}, theta_plus, 8, 0.1, 1, 0);


if nargin < 6
    vis = 0;
    debug = 0;
end

if nargin < 7
    debug = 0;
end

% -- global variables
global q cut_at cut_if

numLeafNodes = size(imgData.adj,1);
numTotalNodes = size(imgTreeTop.kids,1);

if debug
    numTotalNodes = 173; % for allTrees{1}, it's a subtree with 4 leafs
    %numTotalNodes = 224;
end


% -- leafs of each subtree
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


% -- counts of pixel for each subtree
% USE counts 
csp = imgData.labelCountsPerSP; 
% USE conditional likelihood
%csp = imgTreeTop.catOut(:,1:numLeafNodes)' .* repmat(imgData.numPixelInSP, 1, n_labs);

count_csp = sum(csp,2);

% USE P(Y_j | z_j) = Y_jz log(theta_plus / theta_minus) + \sum_{k} Y_jk log(theta_minus)
theta_minus = (1 - theta_plus) ./ (n_labs - 1);
l_theta_plus = log(theta_plus);
l_theta_minus = log(theta_minus);
l_theta_diff = l_theta_plus - l_theta_minus;
csp = csp .* repmat(l_theta_diff, numLeafNodes, 1) ...
    + repmat(count_csp,1,n_labs) .* repmat(l_theta_minus, numLeafNodes, 1);

% -- Q(t) 
q = zeros(numTotalNodes, n_labs); % max likelihood
q(1:numLeafNodes,:) = csp;
% track decisions, values range from 1: merge; 2: cutL; 3: cutR
cut_at = zeros(numTotalNodes,n_labs); 

% DP loop
prior_merge = log(p_connect) + log(p_connect); prior_merge = prior_merge * 5000;
prior_cut = log(p_connect) + log(1-p_connect) - log(n_labs); prior_cut = prior_cut * 5000;
fprintf('p: %f, prior_merge = %f, prior_cut = %f\n', p_connect, prior_merge, prior_cut);

for j = numLeafNodes+1:numTotalNodes
    kids = imgTreeTop.getKids(j);
    L = kids(1); R = kids(2);
    
    [maxL,posL] = max(q(L,:));
    [maxR,posR] = max(q(R,:));
    %maxL = sum(q(L,:));
    %maxR = sum(q(R,:));
    
    q_merge = q(L,:) + q(R,:) + prior_merge;
    q_cutL = maxL + q(R,:) + prior_cut;
    q_cutR = maxR + q(L,:) + prior_cut;
    
    q_all = [q_merge; q_cutL; q_cutR];
    [q(j,:), cut_at(j,:)] = max(q_all);
end


% -- top down decoding
cut_if = zeros(numTotalNodes,1); % if cut above
cut_if(end) = 1; % always cut above root node

j = numTotalNodes;
top_down_decoding(imgTreeTop, j); % output cut_if


% -- superpixel labeling
forest = zeros(numLeafNodes,1); % indicate leafs belong to which tree 
cc = zeros(numTotalNodes, n_labs); % collection of leaf likelihood under resulting tree
%cnt_debug = zeros(numLeafNodes,1); % DEBUG: # of cuts on the path to root

% find the lowest cuts
for i = 1:numLeafNodes
    j = i;
    while 0 ~= j % parent(top) = 0
        if cut_if(j) == 1
            forest(i) = j;
            cc(j,:) = cc(j,:) + csp(i,:); % collect leafs
            break
            %cnt_debug(i) = cnt_debug(i) + 1;
        end
        j = imgTreeTop.pp(j); % go to parent
    end
end

% lemma: for each root in the forest, there is at least one leaf under it
pred_labs = zeros(numLeafNodes,1);
for i = 1:numTotalNodes
    if cut_if(i) == 1 % if it is a root
        [~,l] = max(cc(i,:));
        pred_labs(forest == i) = l;
    end
end


% -- final outputs and visualization
labs = pred_labs;
cuts = cut_if;
%Q = max(q,[],2);
Q = csp(1:numLeafNodes, pred_labs);

if vis
    figure(p_connect*100000);
    colorImgWithLabels(imgData.segs2,imgData.labels,pred_labs,...
            imgData.segLabels, imgData.img);

    sp_map = zeros(size(imgData.labels));
    for i = 1:length(forest)
        sp_map(imgData.segs2 == i) = forest(i);
    end

    figure(p_connect*100000+1);
    imagesc(sp_map);
    title(sprintf('There are %d segments (subtrees) in the forest', sum(cut_if)));
end

%% helper functions
function top_down_decoding(imgTreeTop, j)
% pre-order traversal
global cut_at q cut_if

kids = imgTreeTop.getKids(j);
if kids(1) == 0
    return
end

[~,pos] = max(q(j,:)); % which state is max
decis = cut_at(j,pos); % decision for this state

if decis == 1 % merge
    % so do nothing
elseif decis == 2 % cutL
    cut_if(kids(1)) = 1;
elseif decis == 3 % cutR
    cut_if(kids(2)) = 1;
else
    disp('*** Something Wrong!');
    return
end

top_down_decoding(imgTreeTop, kids(1));
top_down_decoding(imgTreeTop, kids(2));


