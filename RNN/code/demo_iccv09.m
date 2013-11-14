% demo stanford background
%iccv09: 0 void   1,1 sky  0,2 tree   2,3 road  1,4 grass  1,5 water  1,6 building  2,7 mountain 2,8 foreground

clear
close all

% vlfeat
run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');

% load [W, Wbot, Wcat, Wout], params and allData
load ../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
load ../data/iccv09-allData-eval.mat

% load all parse trees
load ../output/iccv09-allTrees-eval.mat

% MLE for theta_plus
MLE_theta;

% example
i = 1;
[Q,cuts,labels] = tree_cut(allData{i}, allTrees{i}, theta_plus, 8, 4, 1, 0);