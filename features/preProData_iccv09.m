% preProData
% reads in images, features, and superpixel information from stephen gould's data

%addpath(genpath('../../toolbox/'));

%dataset = '../data/iccv09/'
%dataFolder = '../data/msrc/';
dataFolder = 'data_iccv09/';

rt_path = 'data_iccv09/Color/';
lb_path = 'data_iccv09/GroundTruth/';
save_path = 'data_iccv09/feat_superpixels/';

evalList = cell(length(allData),1);

for asd=1:1
    
    % evallist: 6000150 has faulty segmentation file, deleted for now :/
    saveFolder = [dataFolder 'allInMatlab/'];
    
    for i = 1:length(allData)
        img = double(allData{i}.img) / 255;
        sizimg = size(img);
        
        labels = double(allData{i}.labels);

        %segs2 = dlmread([dataFolder 'newFeatures/' fileList{i} '.0.seg']);
        %load([save_path, fileList{i}, '_seg.mat'], 'segments');
        %segs2 = segments;
        filename = [];
        [hh, ww, ~] = size(img);
        segs2 = dlmread([save_path, ['iccv09_' num2str(i)], '_sortseg.dat'], ...
            ' ', [1 0 hh ww-1]);
        segs2=segs2+1;
        segs2 = double(segs2);

        assert(min(segs2(:))>0)
        assert(all(size(segs2)==size(labels)))
        
        %feat2= dlmread([dataFolder 'newFeatures/' fileList{i} '.0.txt']);
        load([save_path, ['iccv09_' num2str(i)], '_node_context.mat'], 'H');
        feat2 = H(:,2:end);
        
        assert(size(feat2,1)==max(segs2(:)));
        
        saveName = [saveFolder ['iccv09_' num2str(i)] '.mat'];
        %evalList{i} = saveName;
        if exist(saveName,'file')
            %save(saveName,'img','labels','segs2','segs3','feat2','feat3','-append');
            %save(saveName,'img','labels','segs2','feat2');
            saveName
        else
            %save(saveName,'img','labels','segs2','segs3','feat2','feat3');
            save(saveName,'img','labels','segs2','feat2');
            saveName
        end
        disp(num2str(i))
    end
    
end

for i = 1:length(allData)
  evalList{i} = ['iccv09_' num2str(i) '.mat'];
end

%%%%%%%%%%%%%%%%%%%%%%
% set to 1 if you have <5GB RAM or you just want to see what's going on for debugging/studying
tinyDatasetDebug = 0;

% data set: stanford background data set from Gould et al.
%mainDataSet = 'penn-fudan-all-context';
mainDataSet = 'iccv09';
%setDataFolders
dataSet = 'train';
dataSetEval = 'eval';
dataFolder = 'data_iccv09/allInMatlab/';

rt_path = 'data_iccv09/Color/';
lb_path = 'data_iccv09/GroundTruth/';
save_path = 'data_iccv09/feat_superpixels/';


%%%%%%%%%%%%%%%%%%%%%%%
%iccv09: 0 void   1,1 sky  0,2 tree   2,3 road  1,4 grass  1,5 water  1,6 building  2,7 mountain 2,8 foreground
set(0,'RecursionLimit',1000);
%params.numLabels = 7; % we never predict 0 (void)
params.numLabels = 8;
%params.numFeat = 268;
params.numFeat = 140;


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
neighNameStem = ['data_iccv09/' mainDataSet '-allNeighborPairs'];
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
    %dataSet='train';
    %preProSegFeatsAndSave(dataFolder,neighNameStem,trainList, neighName, dataSet, params,mainDataSet)
    dataSet='train';
    preProSegFeatsAndSave(dataFolder,neighNameStem,evalList, neighNameEval, dataSet, params,mainDataSet)
end
