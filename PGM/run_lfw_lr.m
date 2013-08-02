% This script will train and evaluate a logistic regression
% for the task of Hair/Skin/Background labeling on LFW part label dataset
%
% input (see config_lr.m for default values)
%   rmposfeat   : remove position features from superpixel features
%   verbose     : display progress during testing
%   lrl2reg     : weight decay for node weights
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

function [acc_train, acc_valid, acc_test] = run_lfw_lr(rmposfeat,verbose,lrl2reg)

%%% --- default parameter values --- %%%
config_lr;


%%% --- startup --- %%%
startup_penn_fudan;
olddim = 250;   % original LFW image size
nlabel = 3;     % number of segmentation labels

load('sds_large.mat','sds');
load('esds_large.mat','esds');
if rmposfeat,
    % remove position features
    sds(65:128) = [];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- logistic regression --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('model/lr/');
fname_lr = sprintf('lr_l2r%g_rmposfeat%d',lrl2reg,rmposfeat);
train_lfw_lr;
save(sprintf('%s/%s.mat',fsave_dir,fname_lr),'w_lr');



%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- evaluation --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n=============================\n');
fprintf('Begin testing! (verbose:%d)\n',verbose);
fprintf('=============================\n\n');

acc_train = eval_lfw_lr(w_lr, trainnames, trainnums, sds, verbose);
acc_valid = eval_lfw_lr(w_lr, validnames, validnums, sds, verbose);
acc_test = eval_lfw_lr(w_lr, testnames, testnums, sds, verbose);

fid = fopen(sprintf('%s/lr.txt',log_dir),'a+');
fprintf(fid,'acc (val) = %g, acc (test) = %g, (%s)\n',acc_valid,acc_test,fname_lr);
fclose(fid);

return;
