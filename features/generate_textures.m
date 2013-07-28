%% 
% generate_textures.m
% 
% Code to generate textons and texture features for the following papers : 
%
% 1) Augmenting CRFs with Boltzmann Machine Shape Priors for Image Labeling
% Andrew Kae, Kihyuk Sohn, Honglak Lee and Erik Learned-Miller.
% CVPR, 2013
%
% project page: http://vis-www.cs.umass.edu/GLOC/ 
% 
% 2) Towards unconstrained face recognition
% Gary B. Huang, Manjunath Narayana, and Erik Learned-Miller.
% 
% Workshop on Perceptual Organization in Computer Vision IEEE CVPR, 2008.
% 
% This code requires you to download the Berkeley SegBench from 
% http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/code/segbench.tar.gz
%
% This script is based off code from the SegBench package.
% 
% Modified by Andrew Kae
%

%% Preliminary

startup_directory;

addpath(genpath(segbench_dir));

% Texton filename
texton_file = 'textons.mat';

% Create filterbank
fb = fbCreate(6,1,3,sqrt(2));

% Number of Textons
ntex = 64;

%% Load Textons, otherwise generate them

if exist(texton_file)
    load(texton_file, 'tex');
else

    fprintf('Creating Textons.\n');
    
    %fid = fopen(train_list);
    fid = fopen(sub_list, 'rb');
    names = {};
    nums = [];
    
    %read names and ids
    while(true)
        s = fscanf(fid, '%s', 1);
        if(isempty(s))
            break
        end
        n = fscanf(fid, '%d', 1);
        names{end+1} = s;
        nums(end+1) = n;
    end
    fclose(fid);
    
    % Store the images
    % NOTE : You may want to select a subset of the images instead of all the images.
    im = [];
    
    for i=1:1:numel(nums)
        imfn = sprintf('%s/%s/%s_%04d.jpg', lfw_dir, names{i}, names{i}, nums(i));
        im = [im rgb2gray(double(imread(imfn)) / 255)];
    end    
    
    [tmap,tex] = computeTextons(fbRun(fb, im),ntex);
    
    save(texton_file, 'tex');
end

%% Generate texture features

fprintf('Generating Texture Features\n');

%create texture directory if it doesn't exist
if (~exist(tex_dir, 'dir'))
    mkdir(tex_dir);
end

fid = fopen(all_list);
names = {};
nums = [];

%read names and ids
while(true)
    s = fscanf(fid, '%s', 1);
    if(isempty(s))
        break
    end
    n = fscanf(fid, '%d', 1);
    names{end+1} = s;
    nums(end+1) = n;
end
fclose(fid);

for i=1:numel(nums)
    fprintf('%d\n', i);
    
    imfn = sprintf('%s/%s/%s_%04d.jpg', lfw_dir, names{i}, names{i}, nums(i));
    
    imi = rgb2gray(double(imread(imfn)) / 255);
    
    %run pre-made filters over image at different orientations and resolutions to get
    %responses over the image imi
    fbr =  fbRun(fb, imi);
    
    map = assignTextons(fbr, tex);
    
    %create directory if it doesn't exist
    imdir = [tex_dir names{i}];
    
    if (~exist(imdir, 'dir'))
        mkdir(imdir);
    end
    
    tmfn = sprintf('%s/%s/%s_%04d.dat', tex_dir, names{i}, names{i}, nums(i));
    fidtm = fopen(tmfn, 'w');
    
    fprintf(fidtm, '%d\n', map');
    fclose(fidtm);
end

