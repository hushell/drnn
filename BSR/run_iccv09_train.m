clear all;close all;clc;
addpath lib;
run('/scratch/working/softwares/vlfeat-0.9.16/toolbox/vl_setup.m');
load ../RNN/data/iccv09-allData-train.mat

outDir = 'iccv09data_train/ucm2/';
mkdir(outDir);

tic;
for i =1:numel(allData),
    fprintf('iccv09_train_%d\n', i);
    outFile = fullfile(outDir,['iccv09_train_' num2str(i) '.mat']);
    if exist(outFile,'file'), continue; end
    imgFile=allData{i}.img;
    im2ucm(imgFile, outFile);
end
toc;
