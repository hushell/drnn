% preProData
% reads in images, features, and superpixel information from stephen gould's data

%addpath(genpath('../../toolbox/'));

%dataset = '../data/iccv09/'
%dataFolder = '../data/msrc/';
dataFolder = 'data_weizmann_horse/';

rt_path = 'data_weizmann_horse/Color/';
lb_path = 'data_weizmann_horse/GroundTruth/';
save_path = 'data_weizmann_horse/feat_superpixels/';
files = dir(rt_path);
files = files(1:102); % only used first 100 images
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

for asd=1:2
%     if asd == 2
%         break;
%     end
    if asd==1
        fileList = trainList;
    else
        fileList = evalList;
    end
    
    % evallist: 6000150 has faulty segmentation file, deleted for now :/
    saveFolder = [dataFolder 'allInMatlab/'];
    
    for i = 1:length(fileList)
        imfile = [rt_path, fileList{i}];
        img = double(imread(imfile)) / 255;
        sizimg = size(img);
        
        %labels = dlmread([dataFolder 'labels/' fileList{i} '.regions.txt']);
        labels = imread([lb_path, fileList{i}]);
        labels = imresize(labels, sizimg(1:2));
        labels(labels == 0) = 0;
        labels(labels ~= 0) = 1;
        labels = double(labels);

        %segs2 = dlmread([dataFolder 'newFeatures/' fileList{i} '.0.seg']);
        %load([save_path, fileList{i}, '_seg.mat'], 'segments');
        %segs2 = segments;
        [hh, ww, ~] = size(img);
        segs2 = dlmread([save_path, fileList{i}(1:end-4), '_sortseg.dat'], ...
            ' ', [1 0 hh ww-1]);
        segs2=segs2+1;
        segs2 = double(segs2);

        assert(min(segs2(:))>0)
        assert(all(size(segs2)==size(labels)))
        
        %feat2= dlmread([dataFolder 'newFeatures/' fileList{i} '.0.txt']);
        load([save_path, fileList{i}(1:end-4), '_node_context.mat'], 'H');
        feat2 = H(:,2:end);
        
        assert(size(feat2,1)==max(segs2(:)));
        
        saveName = [saveFolder fileList{i}(1:end-4) '.mat'];
        if exist(saveName,'file')
            %save(saveName,'img','labels','segs2','segs3','feat2','feat3','-append');
            %save(saveName,'img','labels','segs2','feat2');
            saveName
        else
            %save(saveName,'img','labels','segs2','segs3','feat2','feat3');
            %save(saveName,'img','labels','segs2','feat2');
            saveName
        end
        disp(num2str(i))
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%
% set to 1 if you have <5GB RAM or you just want to see what's going on for debugging/studying
tinyDatasetDebug = 0;

% data set: stanford background data set from Gould et al.
%mainDataSet = 'penn-fudan-all-context';
mainDataSet = 'weizmann-horse-100';
%setDataFolders
dataSet = 'train';
dataSetEval = 'eval';
dataFolder = 'data_weizmann_horse/allInMatlab/';

rt_path = 'data_weizmann_horse/Color/';
lb_path = 'data_weizmann_horse/GroundTruth/';
save_path = 'data_weizmann_horse/feat_superpixels/';
files = dir(rt_path);
files = files(1:102); % only used first 100 images
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

%%%%%%%%%%%%%%%%%%%%%%%
%iccv09: 0 void   1,1 sky  0,2 tree   2,3 road  1,4 grass  1,5 water  1,6 building  2,7 mountain 2,8 foreground
set(0,'RecursionLimit',1000);
%params.numLabels = 7; % we never predict 0 (void)
params.numLabels = 2;
%params.numFeat = 268;
params.numFeat = 128;


%%%%%%%%%%%%%%%%%%%%%%
% model parameters (should be ok, found via CV)
params.numHid = 50;
params.regPTC = 0.0001;
params.regC = params.regPTC;
params.LossPerError = 0.05;

%sigmoid activation function:
params.f = @(x) (1./(1 + exp(-x)));
params.df = @(z) (z .* (1 - z));

params.actFunc = 'sigmoid';


%%%%%%%%%%%%%%%%%%%%%%
% input and output file names
neighNameStem = ['data_weizmann_horse/' mainDataSet '-allNeighborPairs'];
if tinyDatasetDebug
    neighName = [neighNameStem '_' dataSet '_tiny.mat'];
else
    neighName = [neighNameStem '_' dataSet '.mat'];
end
neighNameEval = [neighNameStem '_' dataSetEval '.mat'];


%%%%%%%%%%%%%%%%%%%%%%
% load and pre-process training and testing dataset
% the resulting files are provided
if ~exist(neighName,'file')
    %%% first run preProData once for both train and eval!
    dataSet='train';
    preProSegFeatsAndSave(dataFolder,neighNameStem,trainList, neighName, dataSet, params,mainDataSet)
    dataSet='eval';
    preProSegFeatsAndSave(dataFolder,neighNameStem,evalList, neighNameEval, dataSet, params,mainDataSet)
end
