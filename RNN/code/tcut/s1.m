vis = 1;

addpath ../
load ../../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
load ../../data/iccv09-allData-eval.mat
run('/scratch/working/softwares/vlfeat-0.9.16/toolbox/vl_setup.m');

theta_plus = MLE_theta(allData,8);

% [Q1,cuts1,labs1,forest1] = tree_cut_new(allData{4}, allTrees{4}, theta_plus, 8, 0.13, 0, 0);
% [Q,cuts,labs,forest] = tree_cut_postpone(allData{4}, allTrees{4}, theta_plus, 8, 0.13, 0, 0);

i = 5;
topCorr=0;
imgTreeTop = parseImage(topCorr,Wbot,W,Wout,Wcat,allData{i}.adj, ...
                allData{i}.feat2,allData{i}.segLabels,params);
[Q1,cuts1,labs1,forest1] = tree_cut_new(allData{i}, imgTreeTop, theta_plus, 8, 0.12, 1, 0);
figure; imgTreeTop.plotForest(forest1);

