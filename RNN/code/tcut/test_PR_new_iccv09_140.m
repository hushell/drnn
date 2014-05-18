% test PR
vis = 0;
vtree = 0;
dense = 0;

addpath ../
load ../../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
load ../../data/iccv09-allData-eval-140.mat
run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');

% compute all parse trees
tree_file = '../../output/iccv09-allTrees-train-140.mat';
if ~exist('allTrees','var')
    if exist(tree_file,'file')
        load(tree_file); 
    else
        %allTrees = cell(1,length(allData));
        allTrees = cell(1,572);
        %allThres = cell(1,length(allData));
        for i = 1:572
            fprintf('computing %d...\n', i);
            load(['../../../BSR/iccv09data_train/ucm2/iccv09_train_' num2str(i) '.mat']);
            [thisTree,thres_arr] = buildBSRTree2(ucm2,0);
            allTrees{i} = thisTree;
            %allThres{i} = thres_arr;
        end
        save('../../output/iccv09-allTrees-train-140.mat', 'allTrees');
    end
end

theta_plus = MLE_theta(allData,8);

% for each image
PR_peaks = zeros(1,length(allData));
SPAccsMax = zeros(1,length(allData));
p_connect_star = zeros(1,length(allData));
for i = 2:length(allData)
    fprintf('starting %d...\n', i);
  
    if any(sum(allData{i}.labelCountsPerSP,2) == 0)
      PR_peaks(i) = 1;
      SPAccsMax(i) = 1;
      p_connect_star(i) = -1;
      continue
    end
    
    % get which p_connect will be used
    [~,~,~,~,q_max_diff,q_max_diff_ind] = tree_cut_new(allData{i}, allTrees{i}, theta_plus, 8, -1);
    
    p_samples = unique(q_max_diff);
    p_samples = 1./(1+8./exp(p_samples)); % 1 / ( 1 + ( n/exp(q_diff) ) )
    assert(all(p_samples >= 0 & p_samples <= 1));
    p_samples = sort(p_samples(p_samples > 0));
    p_samples = [linspace(0,p_samples(1),10), linspace(p_samples(1),1,5), p_samples, 1/9];
    p_samples = unique(p_samples);
    p_samples = sort(p_samples);
    
    if dense
      p_samples = linspace(0,1,20);
    end
    
    if vis && vtree
      name = ['iccv09/iccv09_' num2str(i) '_'];
    else
      name = [];
    end
    
    fprintf('ongoing %d...\n', i);
    
    [PRs,spAccs,nCuts,PRs3,GCEs,VIs] = evalSegPerImg(name, @tree_cut_new, allData{i}, allTrees{i}, theta_plus, 8, p_samples,vis,vtree);
    combo = (PRs.*spAccs.*PRs3) ./ (nCuts.*(VIs+1e-5));
    %[peak,loc] = max(PRs);
    [peak,loc] = fullmax(combo);
    [mnc,mmi] = min(nCuts(loc));
    PR_peaks(i) = peak(mmi);
    SPAccsMax(i) = max(spAccs);
    p_connect_star(i) = p_samples(loc(mmi));

    fprintf('>>>>>> image %d: best combo value = %f at %f\n', i, peak(mmi), p_samples(loc(mmi)));
    fprintf('------ forest: number of subtrees = %d\n', mnc);
    if vis && isempty(name)
        pause
    end
end