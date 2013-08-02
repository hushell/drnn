% visualizeImageTrees

%if data and parameters are already loaded from trainVRNN
%fullTrainParamName
%load(fullTrainParamName,'Wbot','W','Wout')
%load(fullTrainParamName)

run('/home/hushell/working/deep/vlfeat-0.9.16/toolbox/vl_setup');

if ~exist('visuFolder','var')
    visuFolder = '../output/visualization/';
end

visuFolder = [];

% for i = 1:length(allData)
%     visualizeOneImgMerge(allData{i},Wbot,W,Wout,Wcat,params,visuFolder,i,2)
%     disp(['Done with visualizing image ' num2str(i)]);
% end

for i = 1:length(allData)
    visualizeOneTreeImg(allData{i},Wbot,W,Wout,Wcat,params,visuFolder,i,2)
    disp(['Done with visualizing image ' num2str(i)]);
end

% for i = 1:length(evalSet.allData)
%     visualizeOneTreeImg(evalSet.allData{i},Wbot,W,Wout,Wcat,params,visuFolder,i,2)
%     disp(['Done with visualizing image ' num2str(i)]);
% end



