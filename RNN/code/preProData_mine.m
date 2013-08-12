% preProData
% reads in images, features, and superpixel information from stephen gould's data

%addpath(genpath('../../toolbox/'));

%dataset = '../data/iccv09/'
%dataFolder = '../data/msrc/';
dataFolder = '../data/penn-fudan-all-n1/';

rt_path = '/home/hushell/working/deep/datasets/pedestrian_parsing/Color/';
lb_path = '/home/hushell/working/deep/datasets/pedestrian_parsing/GroundTruth/';
save_path = '/home/hushell/working/deep/datasets/pedestrian_parsing/feat_superpixel/';
files = dir(rt_path);
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
        
        %labels = dlmread([dataFolder 'labels/' fileList{i} '.regions.txt']);
        labels = imread([lb_path, fileList{i}(1:end-4), '_GTnew_index.png']);
        labels(labels == 10) = 1;
        labels(labels == 20) = 2;
        labels(labels == 30) = 3;
        labels(labels == 40) = 4;
        labels(labels == 51) = 5;
        labels(labels == 52) = 5;
        labels(labels == 53) = 5;
        labels(labels == 54) = 5;
        labels(labels == 61) = 6;
        labels(labels == 62) = 6;
        labels(labels == 63) = 6;
        labels(labels == 64) = 6;
        labels(labels == 0) = 7; % BG
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
        load([save_path, fileList{i}(1:end-4), '_node_n1.mat'], 'H');
        feat2 = H(:,2:end);
        
        assert(size(feat2,1)==max(segs2(:)));
        
        saveName = [saveFolder fileList{i}(1:end-4) '.mat'];
        if exist(saveName,'file')
            %save(saveName,'img','labels','segs2','segs3','feat2','feat3','-append');
            save(saveName,'img','labels','segs2','feat2');
        else
            %save(saveName,'img','labels','segs2','segs3','feat2','feat3');
            save(saveName,'img','labels','segs2','feat2');
        end
        disp(num2str(i))
    end
    
end