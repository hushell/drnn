function [Q,cuts,labs,granularity] = merge_cut(imgData, imgTreeTop, n_labs, lambda, vis, debug)
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

global c leafsUnder cut_at pred_labs tree_labs

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
    %maxc(si) = max(c(si,:)) / sum(c(si,:));
    maxc(si) = max(c(si,:));
end

% -- Q(t) == opt(t)
Q = maxc;
Q(numLeafNodes+1:numTotalNodes) = -1;
% different from making change points, in which cut_at(j) = i
% here, cut_at(j) = 1 indicates there is a cut below j
cut_at = zeros(numTotalNodes,1); 

% DP loop
for j = numLeafNodes+1:numTotalNodes
    kids = imgTreeTop.getKids(j);
    Q_st = Q(kids(1)) + Q(kids(2)); assert(Q(kids(1)) > 0 && Q(kids(2)) > 0);
    if maxc(j) < Q_st - lambda % cut
        Q(j) = Q_st - lambda;
        cut_at(j) = 1;
    else % merge
        Q(j) = maxc(j);
    end
end

% -- top down decoding
j = numTotalNodes;
% superpixel labeling
pred_labs = zeros(numLeafNodes,1);
tree_labs = zeros(numTotalNodes,1);

top_down_labeling(imgTreeTop, j);

% -- final outputs
labs = pred_labs;
cuts = cut_at;

% -- granularity
granularity = 0;

for ti = numLeafNodes+1:numTotalNodes
    part_map = zeros(size(imgData.labels));
    l = tree_labs(ti);
    if l > 0
        for li = 1:length(leafsUnder{ti})
            part_map = part_map + imgData.segs2 == leafsUnder{ti}(li);
        end
    end
end

if vis
    %figure;
    colorImgWithLabels(imgData.segs2,imgData.labels,pred_labs,...
            imgData.segLabels, imgData.img);
    pause
end



%% helper functions
function top_down_labeling(imgTreeTop, j)
% pre-order traversal
global c pred_labs leafsUnder cut_at tree_labs

kids = imgTreeTop.getKids(j);
if kids(1) == 0
    [~,l] = max(c(j,:));
    pred_labs(j) = l; % TODO: this can be done by init
    return
end

if cut_at(j) == 1
    top_down_labeling(imgTreeTop, kids(1));
    top_down_labeling(imgTreeTop, kids(2));
else
    [~,l] = max(c(j,:));
    pred_labs(leafsUnder{j}) = l;
    tree_labs(j) = l;
end


