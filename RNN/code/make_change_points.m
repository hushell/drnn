function [Q,cuts,pred_labs] = make_change_points(imgData, imgTreeTop, n_labs, lambda, vis, debug)
% DP Making change points
% Q(t) = max [ c(t), max_{s \in subtree_t} (c(t\s) + Q(s) - lambda) ]
%
% @param imgData
% @param imgTreeTop
% @param lambda penalty for introducing a cut
% @param vis whether visualize the segmentation
%
% @return Q optimal correct labeled pixels for each subtree, Q(end) is the
% solution of DP
% @return cuts above which nodes, cuts occur. NOTE: all zeros means no cuts
% @return pred_labs predicted labels of leafs

if nargin < 5
    vis = 0;
    debug = 0;
end

if nargin < 6
    debug = 0;
end

global S maxS c st_graph

numLeafNodes = size(imgData.adj,1);
numTotalNodes = size(imgTreeTop.kids,1);
if debug
    %numTotalNodes = 173; % for allTrees{1}, it's a subtree with 4 leafs
    numTotalNodes = 224;
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

% total # of pixels in each subtree
count_c = sum(c,2);

% -- complements of cut, one for each (t,s) pair, similar to e_ij pairs
S = zeros(numTotalNodes, numTotalNodes, n_labs);
% # of correct pixels in a complement, NOTE: unnormalized
maxS = ones(numTotalNodes, numTotalNodes) .* -1;
% indicator of which nodes are in the subtree
st_graph = zeros(numTotalNodes, numTotalNodes);

for si = numLeafNodes+1:numTotalNodes
    compute_complement(imgTreeTop, si, si);
end


% -- Q(t) == opt(t)
Q = maxc;
Q(numLeafNodes+1:numTotalNodes) = -1;
cut_at = zeros(numTotalNodes,1);

% DP loop
for j = numLeafNodes+1:numTotalNodes
    cplm = maxS(j,:)';
    nind = find(st_graph(j,:)==1);
    Q_st = Q(nind); 
    assert(min(Q_st) >= 0); % all descendents have been computed
    
    %cplm = (cplm(nind) + Q_st.*count_c(nind)) ./ count_c(j) - lambda;
    cplm = (cplm(nind) + Q_st) - lambda;
    
    %Q(j) = max(maxc(j), max(cplm)); %assert(Q(j) <= 1);
    [max_cplm, i] = max(cplm);
    if maxc(j) >= max_cplm
        Q(j) = maxc(j);
        cut_at(j) = j;
    else
        Q(j) = max_cplm;
        cut_at(j) = nind(i);
    end
end

% -- top down decoding
cuts = zeros(numTotalNodes, 1); % cuts above which nodes
j = numTotalNodes;
% superpixel labeling
pred_labs = zeros(numLeafNodes,1);
[~,l] = max(c(j,:));
pred_labs(:) = l;

while imgTreeTop.kids(j,1) ~= 0 % only internal nodes
    if (cut_at(j) == j)
        break
    end
    
    i = cut_at(j);
    cuts(i) = 1;
    
    [~,lc] = max(S(j,i,:)); assert(lc > 0);
    pred_labs(leafsUnder{j}) = lc; % label complement
    [~,l] = max(c(i,:));
    pred_labs(leafsUnder{i}) = l; % label next subtree
    
    j = i;
end

if vis
    figure;
    run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');
    colorImgWithLabels(imgData.segs2,imgData.labels,pred_labs,...
            imgData.segLabels, imgData.img);
end



%% helper functions
function compute_complement(imgTreeTop, top, rt)
% pre-order traversal
global S maxS c st_graph

if top ~= rt
    st_graph(top, rt) = 1;
end

kids = imgTreeTop.getKids(rt);

if kids(1) == 0
    return
end

S(top, kids(1), :) = c(top,:) - c(kids(1),:);
maxS(top, kids(1)) = max(S(top, kids(1), :));
S(top, kids(2), :) = c(top,:) - c(kids(2),:);
maxS(top, kids(2)) = max(S(top, kids(2), :));

compute_complement(imgTreeTop, top, kids(1));
compute_complement(imgTreeTop, top, kids(2));

