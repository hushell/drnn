
% ckiw script, 23 Jan

vis = 1;

addpath ../
load ../../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
load ../../data/iccv09-allData-eval.mat
run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');

% compute all parse trees
tree_file = '../../output/iccv09-allTrees-eval.mat';
if ~exist('allTrees','var')
    if exist(tree_file,'file')
        load(tree_file); 
    else
        allTrees = cell(1,length(allData));
        for i = 1:length(allData)
            if length(allData{i}.segLabels)~=size(allData{i}.feat2,1)
                disp(['Image ' num2str(i) ' has faulty data, skipping!'])
                continue
            end
            topCorr=0;
            imgTreeTop = parseImage(topCorr,Wbot,W,Wout,Wcat,allData{i}.adj, ...
                allData{i}.feat2,allData{i}.segLabels,params);
            allTrees{i} = imgTreeTop;
        end
        save('../../output/iccv09-allTrees-eval.mat', 'allTrees');
    end
end

theta_plus = MLE_theta(allData,8);

% [Q1,cuts1,labs1,forest1] = tree_cut_new(allData{4}, allTrees{4}, theta_plus, 8, 0.13, 0, 0);
% [Q,cuts,labs,forest] = tree_cut_postpone(allData{4}, allTrees{4}, theta_plus, 8, 0.13, 0, 0);

demo_func(allData, allTrees, theta_plus, 4);
demo_func(allData, allTrees, theta_plus, 5);

