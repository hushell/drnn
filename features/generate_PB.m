%% 
% generate_PB.m
% 
% Code to generate PB features for the following papers : 
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
% This code requires you to download the Superpixel code from 
% http://www.cs.sfu.ca/~mori/research/superpixels/
%
% This script is based off the sp_demo.m code from the superpixel package.
% 
% Modified by Andrew Kae
%

%%

startup_directory;

addpath(genpath(sp_dir));

%% Read in the filenames and ids

names = {};
nums = [];

nsp = 100; nsp2 = 200; nev = 40;

fid = fopen(all_list);

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

%% Generate Pb features

fprintf('Generating Pb\n');

%create PB directory if it doesn't exist
if (~exist(pb_dir, 'dir'))
    mkdir(pb_dir);
end

for i=1:numel(nums)
    fprintf('%d\n', i);
    
    im_file = sprintf('%s/%s/%s_%04d.jpg', lfw_dir, names{i}, names{i}, nums(i));
    
    I = im2double(imread(im_file));
    
    [N,M,tmp] = size(I);
    
    % ncut parameters for superpixel computation
    diag_length = sqrt(N*N + M*M);
    par = imncut_sp;
    par.int=0;
    par.pb_ic=1;
    par.sig_pb_ic=0.05;
    par.sig_p=ceil(diag_length/50);
    par.verbose=0;
    par.nb_r=ceil(diag_length/60);
    par.rep = -0.005;  % stability?  or proximity?
    par.sample_rate=0.2;
    par.nv = nev;
    par.sp = nsp;
    
    % Intervening contour using mfm-pb
    [emag,ephase] = pbWrapper(I,par.pb_timing);
    emag_thick = pbThicken(emag);
    
    %create directories if necessary
    imdir = sprintf('%s/%s/', pb_dir, names{i});
    
    %   if (~exist(imdir, 'dir'))
    if (~exist(imdir))
        mkdir(imdir);
    end
    
    pb_file = sprintf('%s/%s/%s_%04d.emag_thick.txt', pb_dir, names{i}, names{i}, nums(i));
    
    %save emag
    %save(pb_file, 'emag_thick', '-ascii');
    dlmwrite(pb_file, emag_thick, 'delimiter', ' ', 'precision', '%3.10f');
end
