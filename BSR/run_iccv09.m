clear all;close all;clc;
addpath lib;
run('/scratch/working/softwares/vlfeat-0.9.16/toolbox/vl_setup.m');
load ../RNN/data/iccv09-allData-eval.mat

outDir = 'iccv09data/ucm2/';
mkdir(outDir);

tic;
for i =28:numel(allData),
    fprintf('iccv09_%d\n', i);
    outFile = fullfile(outDir,['iccv09_' num2str(i) '.mat']);
    if exist(outFile,'file'), continue; end
    imgFile=allData{i}.img;
    im2ucm(imgFile, outFile);
end
toc;
