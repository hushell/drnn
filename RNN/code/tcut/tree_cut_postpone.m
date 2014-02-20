function [Q,cuts,labs,forest,q_max_diff,q_max_diff_ind,postpone] = ...
    tree_cut_postpone(imgData, imgTreeTop, theta_plus, n_labs, p_connect, vis, debug)
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
% @return forest: forest(j) = i means leaf j is under subtree rooted at i
%
% example: [Q,cuts,labels,forest] = tree_cut(allData{i}, allTrees{i}, theta_plus, 8, 0.1, 1, 0);


if nargin < 6
    vis = 0;
    debug = 0;
end

if nargin < 7
    debug = 0;
end

% -- global variables
global q cut_at cut_if postp

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
% USE counts (GT)
csp = imgData.labelCountsPerSP; 
% USE conditional likelihood
%csp = imgTreeTop.catOut(:,1:numLeafNodes)' .* repmat(imgData.numPixelInSP, 1, n_labs);

count_csp = sum(csp,2);

% compute likelihood for each superpixel
% USE P(Y_j | z_j) = nml * Y_jz log(theta_plus / theta_minus) + \sum_{k} Y_jk log(theta_minus)
theta_minus = (1 - theta_plus) ./ (n_labs - 1);
l_theta_plus = log(theta_plus);
l_theta_minus = log(theta_minus);
l_theta_diff = l_theta_plus - l_theta_minus;
% factorial(n) = gamma(n+1), nml = n!/(x_1!...x_n!)
nml = gammaln(sum(csp,2)+1)-sum(gammaln(csp+1),2);
csp = csp .* repmat(l_theta_diff, numLeafNodes, 1) ...
    + repmat(count_csp,1,n_labs) .* repmat(l_theta_minus, numLeafNodes, 1) ...
    + repmat(nml,1,n_labs);

% -- Q(t) 
q = zeros(numTotalNodes, n_labs); % max likelihood
q(1:numLeafNodes,:) = csp;
% track decisions, values range from 1: merge; 2: cutL; 3: cutR
cut_at = zeros(numTotalNodes,n_labs); 

% DP loop
q_max_diff = zeros(1,numTotalNodes-numLeafNodes);
q_max_diff_ind = numLeafNodes+1:numTotalNodes;

if p_connect < 0
    prior_merge = 0;
    prior_cut = 0;
else
    const1 = 1;
    prior_merge = log(p_connect) + log(p_connect); prior_merge = prior_merge * const1;
    prior_cut = log(p_connect) + log(1-p_connect) - log(n_labs); prior_cut = prior_cut * const1;
    %prior_cut = log(p_connect) + log(1-p_connect); prior_cut = prior_cut * const1 - log(n_labs);
end
fprintf('p: %f, prior_merge = %f, prior_cut = %f, diff = %f\n', ...
    p_connect, prior_merge, prior_cut, prior_merge-prior_cut);

fp = fopen('equally-good.txt', 'a');
fprintf(fp, '=========== p_connet = %f ==========\n', p_connect);

for j = numLeafNodes+1:numTotalNodes
    kids = imgTreeTop.getKids(j);
    L = kids(1); R = kids(2);
    
    [maxL,posL] = max(q(L,:));
    [maxR,posR] = max(q(R,:));
    %maxL = sum(q(L,:));
    %maxR = sum(q(R,:));
    
    %prior_merge = 0.1;
    %prior_cut = 0;
    
    q_merge = q(L,:) + q(R,:) + prior_merge;
    q_cutL = maxL + q(R,:) + prior_cut;
    q_cutR = maxR + q(L,:) + prior_cut;
    
    q_all = [q_merge; q_cutL; q_cutR];
    [q(j,:), cut_at(j,:)] = max(q_all);
    
    fprintf('------------ node %d -------------\n', j);
    
    fprintf('q_L %d = \n', L); disp(q(L,:));
    fprintf('q_R %d = \n', R); disp(q(R,:));
%     
%     q_diff = q_merge - q(j,:);
    q_all
    fprintf('q_j %d = \n', j); disp(q(j,:));
    fprintf('cut_at_where %d = \n', j); disp(cut_at(j,:));
%     q_diff

q_max_diff(j-numLeafNodes) = max(max(q_all(2:end,:))) - max(q_all(1,:));

q_diff_ind_1 = find(q_merge - q_cutL == 0);
q_diff_ind_2 = find(q_merge - q_cutR == 0);

if ~isempty(intersect(q_diff_ind_1, q_diff_ind_2)) && p_connect > 0
    fprintf('!!! equally good !!!\n\n\n');
    fprintf(fp, '------------ equally good node %d -------------\n', j);
    fprintf(fp, 'q_L %d = \n', L); fprintf(fp, '%f ', q(L,:)); fprintf(fp, '\n');
    fprintf(fp, 'q_R %d = \n', R); fprintf(fp, '%f ', q(R,:)); fprintf(fp, '\n');
    fprintf(fp, 'q_all = \n');
    fprintf(fp, '%f ', q_all(1,:)); fprintf(fp, '\n');
    fprintf(fp, '%f ', q_all(2,:)); fprintf(fp, '\n');
    fprintf(fp, '%f ', q_all(3,:)); fprintf(fp, '\n');
    fprintf(fp, 'q_j %d = \n', j); fprintf(fp, '%f ', q(j,:)); fprintf(fp, '\n');
    fprintf(fp, 'cut_at_where %d = \n', j); fprintf(fp, '%f ', cut_at(j,:)); fprintf(fp, '\n');
    [tmp_max, tmp_ind] = max(q(j,:));
    fprintf(fp, 'max state = %f %d \n', tmp_max, tmp_ind);
    fprintf(fp, '\n');
end
end

fclose(fp);

if p_connect < 0
    [q_max_diff, tmp_ind] = sort(q_max_diff);
    q_max_diff_ind = q_max_diff_ind(tmp_ind);
end


% -- top down decoding
cut_if = zeros(numTotalNodes,1); % if cut above
cut_if(end) = 1; % always cut above root node
postp = zeros(numTotalNodes,1); % if postpone the decision of cut which kid

j = numTotalNodes;
% NOTE if a decision is postponed, both kids will be cut at this moment
% i.e., treat it as a leaf node currently
top_down_decoding(imgTreeTop, j); % output cut_if


% -- superpixel labeling
forest = zeros(numTotalNodes,1); % indicate leafs belong to which tree 
cc = zeros(numTotalNodes, n_labs); % collection of leaf likelihood under resulting tree
%cnt_debug = zeros(numLeafNodes,1); % DEBUG: # of cuts on the path to root

% find the lowest cuts
for i = 1:numTotalNodes
    j = i;
    while 0 ~= j % parent(top) = 0
        if cut_if(j) == 1 
            forest(i) = j;
            if i <= numLeafNodes
                cc(j,:) = cc(j,:) + csp(i,:); % collect leafs
            end
            break
            %cnt_debug(i) = cnt_debug(i) + 1;
        end
        j = imgTreeTop.pp(j); % go to parent
    end
end

% labeling 
% NOTE postponed nodes will be labeled indepently of its kids in this phase
% lemma: for each root in the forest, there is at least one leaf under it
pred_labs = zeros(numTotalNodes,1);
for i = 1:numTotalNodes
    if cut_if(i) == 1 % if it is a root
        [maxcc,l] = max(cc(i,:));
        if maxcc == 0
            l = 0;
        end
        pred_labs(forest == i) = l;
    end
end

% renew cut_if for those postponed nodes
% basic idea: cut will be canceled if kid has same label as its parent
for i = numTotalNodes:-1:1
    if postp(i) == 1
        kids = imgTreeTop.getKids(i);
        L = kids(1); R = kids(2);
        labelL = pred_labs(L);
        labelR = pred_labs(R);
        labelP = pred_labs(forest(i));
        
        % seems only need to change cut_if, shall relabel?
        if labelL == labelP
            cut_if(L) = 0;
        else
            cut_if(R) = 0;
        end
    end
end

% -- after updating cut_if, we do the labeling procedure again
forest = zeros(numLeafNodes,1); % indicate leafs belong to which tree 
cc = zeros(numTotalNodes, n_labs); % collection of leaf likelihood under resulting tree

% for i = 1:numLeafNodes
%     j = i;
%     while 0 ~= j % parent(top) = 0
%         if cut_if(j) == 1 
%             forest(i) = j;
%             cc(j,:) = cc(j,:) + csp(i,:); % collect leafs
%             break
%         end
%         j = imgTreeTop.pp(j); % go to parent
%     end
% end

for i = 1:numTotalNodes
    j = i;
    while 0 ~= j % parent(top) = 0
        if cut_if(j) == 1 
            forest(i) = j;
            if i <= numLeafNodes
                cc(j,:) = cc(j,:) + csp(i,:); % collect leafs
            end
            break
        end
        j = imgTreeTop.pp(j); % go to parent
    end
end

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
postpone = postp;

if vis
    figure(p_connect*100000);
    %figure(100);
    colorImgWithLabels_vlfeat(imgData.segs2,imgData.labels,pred_labs,...
            imgData.segLabels, imgData.img);

    sp_map = zeros(size(imgData.labels));
    for i = 1:length(forest)
        sp_map(imgData.segs2 == i) = forest(i);
    end

    figure(p_connect*100000+1);
    %figure(101);
    imagesc(sp_map);
    title(sprintf('There are %d segments (subtrees) in the forest', sum(cut_if)));
end

%% helper functions
function top_down_decoding(imgTreeTop, j)
% pre-order traversal
global cut_at q cut_if postp

kids = imgTreeTop.getKids(j);
if kids(1) == 0
    return
end

[mq,pos] = max(q(j,:)); % which state is max
decis = cut_at(j,pos); % decision for this state

loc = q(j,:) == mq;
other_decis = cut_at(j,loc); % look for other decision that has equal q

if ~isempty(find(other_decis ~= decis, 1))
    postp(j) = 1;
    cut_if(kids(1)) = 1;
    cut_if(kids(2)) = 1;
else
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
end

top_down_decoding(imgTreeTop, kids(1));
top_down_decoding(imgTreeTop, kids(2));


