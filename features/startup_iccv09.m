%%% --- data directories --- %%%
data_dir = 'data_iccv09';

%% List of files to process

all_list = 'parts_all.txt';
train_list = 'parts_train.txt';
sub_list = 'parts_sub.txt';

%% These directories should already exist

lfw_dir = [data_dir '/Color/'];

gt_dir = [data_dir '/GroundTruth/'];

%% These directories will be created

% textures
tex_dir = [data_dir '/parts_tex/'];

% Pb 
pb_dir = [data_dir '/parts_pb/'];

%% External packages

% directory containing the SegBench code
% http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/code/segbench.tar.gz
segbench_dir = './segbench/';

% directory containing superpixel code
% http://www.cs.sfu.ca/~mori/research/superpixels/
sp_dir = './superpixels64/';

spfeat_dir = [data_dir '/parts_spseg_features/'];

%% save images and GTs
load ../RNN/data/iccv09-allData-train.mat

%% NOTE: run the following separately if you don't want to generate these everytime calling this script
% for i = 1:length(allData)
%   im = allData{i}.img;
% %   gt = allData{i}.labels;
%   imwrite(im, [lfw_dir 'iccv09_' num2str(i) '.jpg']);
% %   imwrite(gt, [gt_dir 'iccv09_' num2str(i) '.png']);
% end
% 
% fp = fopen([data_dir '/list.txt'], 'w');
% for i = 1:length(allData)
%   filename = ['iccv09_' num2str(i)];
%   siz = size(allData{i}.labels);
%   fprintf(fp, '%s %d %d\n', filename, siz(1), siz(2));
% end
% fclose(fp);
