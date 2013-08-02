%%% --- data directories --- %%%
data_dir = '../features/data_penn_fudan';
rt_dir = [data_dir '/Color/'];

%%% List of files to process

% all_list = 'parts_all.txt';
% train_list = 'parts_train.txt';
% sub_list = 'parts_sub.txt';

%%% --- addpath --- %%%
%addpath data/;
addpath minFunc/;
addpath utils/;
addpath common/;

%%% --- directory to store model parameters --- %%%
fsave_dir = 'weights';
if ~exist(fsave_dir,'dir'),
    mkdir(fsave_dir);
end

%%% --- directory for log --- %%%
log_dir = 'log';
if ~exist(log_dir,'dir'),
    mkdir(log_dir);
end

%% training and testing files
files = dir(rt_dir);
fileList = cell(length(files)-2,1);
tri = 1; evi = 1;
for j = 3:length(files)
    fileList{j-2} = files(j).name;
end

trainList = cell(tri-1,1);
evalList = cell(evi-1,1);
tri = 1; evi = 1;
for j = 1:numel(fileList)
    if mod(j,2) ~= 0
        trainList{tri} = fileList{j};
        tri = tri + 1;
    else
        evalList{evi} = fileList{j};
        evi = evi + 1;
    end
end
num_train = numel(trainList);
num_test = numel(evalList);

