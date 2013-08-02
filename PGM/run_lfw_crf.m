% This script will train and evaluate conditional random fields
% for the task of Hair/Skin/Background labeling on LFW part label dataset
%
% input (see config_crf.m for default values)
%   rmposfeat   : remove position features from superpixel features
%   verbose     : display progress during testing
%   lrl2reg     : weight decay for node weights
%   l2reg_node  : weight decay for node weights
%   l2reg_edge  : weight decay for edge weights
%
% output
%   acc_train   : training accuracy
%   acc_valid   : validation accuracy
%   acc_test    : testing accuracy
%
%
% reference:
% Augmenting CRFs with Boltzmann Machine Shape Priors for Image Labeling, CVPR, 2013.
%

function [acc_train, acc_valid, acc_test] = run_lfw_crf(rmposfeat,verbose,lrl2reg,l2reg_node,l2reg_edge)

%%% --- default parameter values --- %%%
config_crf;


%%% --- startup --- %%%
startup;
olddim = 250;   % original LFW image size
nlabel = 3;     % number of segmentation labels

load('sds_large.mat','sds');
load('esds_large.mat','esds');
if rmposfeat,
    sds(65:128) = [];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- logistic regression --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('model/lr/');
fname_lr = sprintf('lr_l2r%g_rmposfeat%d',lrl2reg,rmposfeat);
train_lfw_lr;
save(sprintf('%s/%s.mat',fsave_dir,fname_lr),'w_lr');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- conditional random field          --- %%%
%%% --- with mean-field inference         --- %%%
%%% --- train edge weights first, and     --- %%%
%%% --- joint train node and edge weights --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('model/crf/');
fsave_dir = sprintf('%s/%s/',fsave_dir,fname_lr);
if ~exist(fsave_dir,'dir'),
    mkdir(fsave_dir);
end
fname_crf = sprintf('crf_l2n%g_l2e%g_rmposfeat%d',l2reg_node,l2reg_edge,rmposfeat);
train_lfw_crf;
save(sprintf('%s/%s.mat',fsave_dir,fname_crf),'w_crf');



%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- evaluation --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n=============================\n');
fprintf('Begin testing! (verbose:%d)\n',verbose);
fprintf('=============================\n\n');

acc_train = eval_lfw_crf(w_crf, trainnames, trainnums, sds, esds, verbose);
acc_valid = eval_lfw_crf(w_crf, validnames, validnums, sds, esds, verbose);
acc_test = eval_lfw_crf(w_crf, testnames, testnums, sds, esds, verbose);

fid = fopen(sprintf('%s/crf.txt',log_dir),'a+');
fprintf(fid,'acc (val) = %g, acc (test) = %g, (%s)\n',acc_valid,acc_test,fname_crf);
fclose(fid);

return;
