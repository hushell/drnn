%% Compute globalPb and hierarchical segmentation for an example image.

addpath(fullfile(pwd,'lib'));

%% 1. compute globalPb on a small image to test mex files
clear all; close all; clc;

imgFile = 'data/0011116.jpg';
outFile = 'data/0011116_gPb.mat';

gPb_orient = globalPb(imgFile, outFile);
delete(outFile);

%figure; imshow(max(gPb_orient,[],3)); colormap(jet);

ucm = contours2ucm(gPb_orient, 'imageSize');
%figure;imshow(ucm);
