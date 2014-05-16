function [thisTree,thres_arr] = buildBSRTree(ucm2,k0,step)
%load data/101087_ucm2.mat % ucm2
%load data/101087_gt.mat % groundTruth
if nargin < 3
  k0 = 0.4; % initial threshold
  step = 0.01; % stride
end

ucm = ucm2(3:2:end, 3:2:end);
%thres_all = k0:step:1.0;
thres_all = sort(unique(ucm(:)))';
len = numel(thres_all);
n_seg_old = numel(setdiff(unique(bwlabel(ucm <= k0)),0));

%run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m')

thres_arr = []; m = 1;
for k = thres_all
    n_seg = numel(setdiff(unique(bwlabel(ucm <= k)),0));
    if n_seg < n_seg_old % check if new threshold results less num of segs
        if n_seg < n_seg_old -1 && n_seg > 0 % TODO: the case >= 3 segs merge together
            fprintf(['threshod = %f, n_seg = %f, n_seg_old = %f: ' ...
              'fail to construct binary tree!\n'], k, n_seg, n_seg_old);
            break
        end
        n_seg_old = n_seg;
        thres_arr(m) = k;
        m = m + 1;
    end
    fprintf('threshold: %f, n_seg = %d\n', k, n_seg);
    %vl_tightsubplot(len,m-1,'box','outer'); imagesc(bwlabel(ucm <= k)); title(sprintf('%f',k)); axis off
end
% thres_arr(m) = 1.0;
% thres_arr = [k0 thres_arr];
assert(n_seg == 1);

% construct tree
init_map = bwlabel(ucm <= k0);
sp_id = unique(init_map);
sp_id = sp_id(sp_id > 0); % 0 is boundary
numTotalSegs = length(sp_id);
numTotalSuperSegs = numTotalSegs+numTotalSegs-1;
sup_nod_id = numTotalSegs + 1;
assert(all([1:numTotalSegs]' == unique(sp_id)));

thisTree = tree();
thisTree.pp = zeros(numTotalSuperSegs,1); % we have numRemSegs many leaf nodes and numRemSegs-1 many nonterminals
thisTree.kids = zeros(numTotalSuperSegs,2);
% thisTree.numLeafNodes = numTotalSegs; 

% PRI = zeros(size(thres_arr));
% VI = zeros(size(thres_arr));
% GCE = zeros(size(thres_arr));

for i = 1:length(thres_arr)
    fprintf('threshold: %d:%f\n', i, thres_arr(i));
    k = thres_arr(i);
    lab_map = bwlabel(ucm <= k);
    
    n_r = numel(unique(lab_map));
    fprintf('# of labels = %d\n', n_r-1);
    assert(all([1:n_r-1]' == setdiff(unique(lab_map),0))); % assert bwlabel return continuous labels
    
    merge_flag = false;
    for j = 1:n_r
        corres_map = init_map(lab_map == j);
        involve_nods = unique(corres_map);
        involve_nods = involve_nods(involve_nods > 0);
        if numel(involve_nods) == 2
            fprintf('merge nodes: '); disp(involve_nods);
            thisTree.pp(involve_nods(1)) = sup_nod_id;
            thisTree.pp(involve_nods(2)) = sup_nod_id;
            thisTree.kids(sup_nod_id,:) = involve_nods';
            init_map(lab_map == j) = sup_nod_id;
            sup_nod_id = sup_nod_id + 1;
            merge_flag = true;
            break
        end
    end
    assert(merge_flag == true);
    % eval
    %[PRI(i) VI(i) ] = match_segmentations2(lab_map+1, groundTruth);
end

%subplot(1,2,1); plot(thres_arr(2:end), PRI(2:end), '-go', 'linewidth', 2); legend('PRI');
%subplot(1,2,2); plot(thres_arr(2:end), VI(2:end), '-bo', 'linewidth', 2); legend('VI');

% function add_parent(thisTree, involve_nods)
% thisTree.pp(involve_nods(1)) = sup_nod_id;
% thisTree.pp(involve_nods(2)) = sup_nod_id;
% thisTree.kids(sup_nod_id,:) = involve_nods';
