function [thisTree,thres_arr] = buildBSRTree2(ucm2,k0)
%load data/101087_ucm2.mat % ucm2
%load data/101087_gt.mat % groundTruth
vis = 0;

ucm = ucm2(3:2:end, 3:2:end);
thres_all = sort(unique(ucm(:)))';
len = numel(thres_all);

%run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m')

thres_arr = thres_all;
if nargin < 2
  k0 = thres_arr(1); % initial threshold
end

% construct tree
init_map = bwlabel(ucm <= k0, 4);
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

for i = 2:length(thres_arr)
    %fprintf('threshold: %d:%f\n', i, thres_arr(i));
    k = thres_arr(i);
    lab_map = bwlabel(ucm <= k, 4);
    
    n_r = numel(unique(lab_map));
    %fprintf('# of labels = %d\n', n_r-1);
    assert(all([1:n_r-1]' == setdiff(unique(lab_map),0))); % assert bwlabel return continuous labels
    
    %merge_flag = false;
    for j = 1:n_r
        corres_map = init_map(lab_map == j); % corres nodes in init_map
        involve_nods = unique(corres_map);
        involve_nods = involve_nods(involve_nods > 0);
        
        while numel(involve_nods) >= 2
            %fprintf('merge nodes: %d %d\n', involve_nods(1),involve_nods(2));
            if numel(involve_nods) > 2
              fprintf('interesting!\n');
            end
            thisTree.pp(involve_nods(1)) = sup_nod_id;
            thisTree.pp(involve_nods(2)) = sup_nod_id;
            thisTree.kids(sup_nod_id,:) = involve_nods(1:2)';
            %init_map(lab_map == j) = sup_nod_id;
            joint_loc = init_map == involve_nods(1) | init_map == involve_nods(2);
            init_map(joint_loc) = sup_nod_id;
            involve_nods = [sup_nod_id; involve_nods(3:end)];
            sup_nod_id = sup_nod_id + 1;
            %merge_flag = true;
            
            if vis
              %imagesc(init_map == sup_nod_id - 1);
              imagesc(init_map > numTotalSegs);
              pause
            end
        end
    end
    %assert(merge_flag == true);
    % eval
    %[PRI(i) VI(i) ] = match_segmentations2(lab_map+1, groundTruth);
end
assert(sup_nod_id == numTotalSuperSegs+1);
%subplot(1,2,1); plot(thres_arr(2:end), PRI(2:end), '-go', 'linewidth', 2); legend('PRI');
%subplot(1,2,2); plot(thres_arr(2:end), VI(2:end), '-bo', 'linewidth', 2); legend('VI');

% function add_parent(thisTree, involve_nods)
% thisTree.pp(involve_nods(1)) = sup_nod_id;
% thisTree.pp(involve_nods(2)) = sup_nod_id;
% thisTree.kids(sup_nod_id,:) = involve_nods';
