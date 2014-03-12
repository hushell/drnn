% preProData
% reads in images, features, and superpixel information from stephen gould's data

%addpath(genpath('../../toolbox/'));

%dataset = '../data/iccv09/'
%dataFolder = '../data/msrc/';
dataFolder = '../../features/data_weizmann_horse/';

rt_path = '../../features/data_weizmann_horse/Color/';
lb_path = '../../features/data_weizmann_horse/GroundTruth/';
save_path = '../../features/data_weizmann_horse/feat_superpixels/';
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
            save(saveName,'img','labels','segs2','feat2');
        else
            %save(saveName,'img','labels','segs2','segs3','feat2','feat3');
            save(saveName,'img','labels','segs2','feat2');
        end
        disp(num2str(i))
    end
    
end