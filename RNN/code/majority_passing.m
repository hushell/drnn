function [Q,cuts,labs] = majority_passing(imgData, imgTreeTop, n_labs, lambda, vis, debug)
% DP: merge or cut, specifically,
% Q(t) = max [ c(t), max_{s = kids(t)} Q(s) - lambda ]
%
% @param imgData: image related data
% @param imgTreeTop: RNN tree
% @param lambda: penalty for introducing a cut
% @param vis: whether visualize the segmentation
% NOTE: for visualizing, you need to have vlfeat somewhere and setup it, e.g.
% run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');
%
% @return Q: optimal correct labeled pixels for each subtree, Q(end) is the
% solution of DP
% @return cuts: cuts(j) = 1 means there is a cut below j, not necessarily to
% know the cut in left branch or right branch
% @return pred_labs: predicted labels of leafs
%
% example:
% for i = 1:length(allData)
%     [Q,cuts,labels] = merge_cut(allData{i}, allTrees{i}, 7, 0, 1, 0);
% end

if nargin < 5
    vis = 0;
    debug = 0;
end

if nargin < 6
    debug = 0;
end

global q leafsUnder cut_at pred_labs

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
c = zeros(numTotalNodes, n_labs);
% purity: # of correctly labeled pixels / count
maxc = zeros(numTotalNodes, 1);

for si = 1:numTotalNodes
    for li = 1:length(leafsUnder{si})
        c(si,:) = c(si,:) + imgData.labelCountsPerSP(leafsUnder{si}(li),:);
    end
    maxc(si) = max(c(si,:));
end
% total # of pixels in each subtree
count_c = sum(c,2);

% -- Q(t) 
q = c;
q(numLeafNodes+1:end,:) = -1;

cut_at = zeros(numTotalNodes,2); 

% DP loop
for j = numLeafNodes+1:numTotalNodes
    kids = imgTreeTop.getKids(j);
    L = kids(1); R = kids(2);
    
    [maxL,posL] = max(q(L,:));
    [maxR,posR] = max(q(R,:));
    
    q_aug_L = q(L,:);
    q_aug_R = q(R,:);
    q_aug_L(posL) = q_aug_L(posL) + maxR - lambda;
    q_aug_R(posR) = q_aug_R(posR) + maxL - lambda;
    
    q_0 = q(L,:) + q(R,:); q_0_m = max(q_0);
    q_cut_L = q_aug_R; q_cut_L_m = max(q_cut_L);
    q_cut_R = q_aug_L; q_cut_R_m = max(q_cut_R);
    
    if q_0_m >= max(q_cut_L_m, q_cut_R_m)
        q(j,:) = q_0;
    elseif (q_cut_L_m > q_cut_R_m) ...
            || (q_cut_L_m == q_cut_R_m && count_c(L) >= count_c(R))
        q(j,:) = q_cut_L;
        cut_at(j,1) = 1;
    else
        q(j,:) = q_cut_R;
        cut_at(j,2) = 1;
    end
end

% -- top down decoding
j = numTotalNodes;
% superpixel labeling
pred_labs = zeros(numLeafNodes,1);
[~,l] = max(q(j,:));
pred_labs(:) = l;

top_down_labeling(imgTreeTop, j);

% -- final outputs
labs = pred_labs;
cuts = sum(cut_at,2);
Q = max(q,[],2);

if vis
    %figure;
    colorImgWithLabels(imgData.segs2,imgData.labels,pred_labs,...
            imgData.segLabels, imgData.img);
end



%% helper functions
function top_down_labeling(imgTreeTop, j)
% pre-order traversal
global pred_labs leafsUnder cut_at q

kids = imgTreeTop.getKids(j);
if kids(1) == 0
    return
end

if cut_at(j,1) == 1
    [~,l] = max(q(kids(1),:));
    pred_labs(leafsUnder{kids(1)}) = l;
elseif cut_at(j,2) == 1
    [~,l] = max(q(kids(2),:));
    pred_labs(leafsUnder{kids(2)}) = l;
end

top_down_labeling(imgTreeTop, kids(1));
top_down_labeling(imgTreeTop, kids(2));


