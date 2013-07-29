%%% --- data directories --- %%%
data_dir = 'data_penn_fudan';

%% List of files to process

all_list = 'parts_all.txt';
train_list = 'parts_train.txt';
sub_list = 'parts_sub.txt';

%% These directories should already exist

lfw_dir = [data_dir '/Color/'];

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
