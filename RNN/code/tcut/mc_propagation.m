function [Q,cuts,labs] = mc_propagation(imgData, imgTreeTop, n_labs, lambda, vis, debug)
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
%     [Q,cuts,labels] = mc_propagation(allData{i}, allTrees{i}, 7, 0, 1, 0);
% end

if nargin < 5
    vis = 0;
    debug = 0;
end

if nargin < 6
    debug = 0;
end

global q leafsUnder cut_at pred_labs c cut_if

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

% USE counts or conditional likelihood
%csp = imgData.labelCountsPerSP;
csp = imgTreeTop.catOut';

for si = 1:numTotalNodes
    for li = 1:length(leafsUnder{si})
        c(si,:) = c(si,:) + csp(leafsUnder{si}(li),:);
    end
    maxc(si) = max(c(si,:));
end
% total # of pixels in each subtree
count_c = sum(c,2);

% -- Q(t) 
q = c;
q(numLeafNodes+1:end,:) = -1;
% 1: purify; 2: merge; 3: cutL; 4: cutR
cut_at = zeros(numTotalNodes,n_labs); 

% DP loop
for j = numLeafNodes+1:numTotalNodes
    kids = imgTreeTop.getKids(j);
    L = kids(1); R = kids(2);
    
    [maxL,posL] = max(q(L,:));
    [maxR,posR] = max(q(R,:));
    
    q_pure = c(j,:);
    q_merge = q(L,:) + q(R,:);
    q_cutL = maxL - lambda + q(R,:);
    q_cutR = maxR - lambda + q(L,:);
    %q_cutL = q(R,:); q_cutL(posR) = q_cutL(posR) + maxL - lambda;
    %q_cutR = q(L,:); q_cutR(posL) = q_cutR(posL) + maxR - lambda;
    %q_all = [q_pure; q_merge; q_cutL; q_cutR];
    q_all = [q_merge; q_cutL; q_cutR];
    
    [q(j,:), cut_at(j,:)] = max(q_all);
end

% -- top down decoding
j = numTotalNodes;
% superpixel labeling
pred_labs = zeros(numLeafNodes,1);
% [~,l] = max(q(j,:));
% pred_labs(:) = l;

cut_if = zeros(numTotalNodes,1); % if cut above
cut_if(end) = 1;

top_down_decoding(imgTreeTop, j);

% -- labeling
forest = zeros(numLeafNodes,1); 
cnt_debug = zeros(numLeafNodes,1); % DEBUG: # cuts on the path to root
cc = zeros(numTotalNodes, n_labs);
for i = 1:numLeafNodes
    j = i;
    while 0 ~= j
        if cut_if(j) == 1
            forest(i) = j;
            cc(j,:) = cc(j,:) + csp(i,:);
            break
            %cnt_debug(i) = cnt_debug(i) + 1;
        end
        j = imgTreeTop.pp(j);
    end
end

for i = 1:numTotalNodes
    if cut_if(i) == 1
        [~,l] = max(cc(i,:));
        pred_labs(forest == i) = l;
    end
end

% can we use Q for labeling? seems yes, but why?
% for i = numTotalNodes:-1:1
%     if cut_if(i) == 1
%         [~,l] = max(q(i,:));
%         pred_labs(leafsUnder{i}) = l;
%     end
% end

% -- final outputs
labs = pred_labs;
cuts = cut_if;
Q = max(q,[],2);

if vis
    %figure;
    colorImgWithLabels(imgData.segs2,imgData.labels,pred_labs,...
            imgData.segLabels, imgData.img);
end



%% helper functions
function top_down_decoding(imgTreeTop, j)
% pre-order traversal
global cut_at q cut_if

kids = imgTreeTop.getKids(j);
if kids(1) == 0
    return
end

[~,pos] = max(q(j,:));
decis = cut_at(j,pos);

decis = decis + 1;
if decis == 1 % purify
    return
elseif decis == 2 % merge
    % so do nothing
elseif decis == 3 % cutL
    cut_if(kids(1)) = 1;
elseif decis == 4 % cutR
    cut_if(kids(2)) = 1;
else
    disp('*** Something Wrong!');
    return
end

top_down_decoding(imgTreeTop, kids(1));
top_down_decoding(imgTreeTop, kids(2));


