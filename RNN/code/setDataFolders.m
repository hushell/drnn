%setDataFolders

dataFolder = ['../data/' mainDataSet '/allInMatlab/'];
analysisFile = '../output/analysis.txt';
analysisFileFull = '../output/analysisPixelAcc_RELEASE.txt';

visuFolder = '../output/visualization/';
dataSet = 'train';
dataSetEval = 'eval';

% % if isunix
% %     disp('Full dataset on UNIX')
% trainList = readTextFile(['../data/' mainDataSet '/' dataSet 'List.txt']);
% evalList = readTextFile(['../data/' mainDataSet '/'  dataSetEval 'List.txt']);
% % else
% %     disp('Debug on Windows')
% %     trainList = readTextFile(['../data/' mainDataSet '/' dataSet 'ListDEBUG.txt']);
% %     evalList = readTextFile(['../data/' mainDataSet '/' dataSetEval 'ListDEBUG.txt']);
% % end

rt_path = '/home/hushell/working/deep/datasets/pedestrian_parsing/Color/';
lb_path = '/home/hushell/working/deep/datasets/pedestrian_parsing/GroundTruth/';
save_path = '/home/hushell/working/deep/datasets/pedestrian_parsing/feat_superpixel/';
files = dir(rt_path);
fileList = cell(length(files)-2,1);
tri = 1; evi = 1;
for j = 3:length(files)
    fileList{j-2} = files(j).name;
end

% trainList = fileList;
% evalList = fileList;
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
