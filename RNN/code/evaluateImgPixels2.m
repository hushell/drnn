function [acc mpr] = evaluateImgPixels2(allData,Z)
addpath ~/working/deep/netlab3_3/
%Z = cell(1,length(allData));

allPR = zeros(1,length(allData));
allCorrectPixels = 0;
allPixels = 0;
for i = 1:length(allData)
    if length(allData{i}.segLabels)~=size(allData{i}.feat2,1) ...
        || numel(unique(unique(allData{i}.segs2))) ~= length(allData{i}.segLabels) ...
        || length(allData{i}.segLabels) == 1 
        disp(['Image ' num2str(i) ' has faulty data, skipping!'])
        continue
    end
    [correctPixels totalPixelsImg allPR(i)] = evaluateOneImgPixels(allData{i},Z{i}');
    allCorrectPixels = allCorrectPixels + correctPixels ;
    allPixels = allPixels + totalPixelsImg;
    if mod(i,10)==0
        disp(['Done with computing image ' num2str(i)]);
    end
end
acc = allCorrectPixels/allPixels;
mpr = mean(allPR);
