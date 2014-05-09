function [correctPixels totalPixels pr] = evaluateOneImgPixels(imgData,catOut)

n_labs = size(imgData.labelCountsPerSP,2);
numLeafNodes = size(imgData.adj,1);
outImg = zeros(size(imgData.segs2,1),size(imgData.segs2,2));

for s = 1:numLeafNodes
	if size(catOut,2) == 1
		thisSegLabel = catOut(s);
	else
    	%finalLabelProbs = catOut(:,s);
    	finalLabelProbs = catOut(s,:);
    	% collect all parent indices
    	
    	[~,thisSegLabel]= max(finalLabelProbs);
	end
    outImg(imgData.segs2==s) = thisSegLabel;
end

correctTestImg = outImg==imgData.labels;
correctPixels = sum(correctTestImg(:));
% ignore 0 = void labels in total count (like Gould et al.)
% (we never predict 0 either)
totalPixels = sum(sum(imgData.labels>0));

% RI value
connectCompGT = zeros(size(imgData.segs2));
spLabelGT = zeros(size(imgData.segs2));

% get SP GT
for i = 1:length(imgData.segLabels)
    spLabelGT(imgData.segs2 == i) = imgData.segLabels(i);
end

% get connected component label
clabs = 1:300;
cci = 1;
for i = 1:n_labs
    %temp = bwlabel(imgData.labels == i); 
    temp = bwlabel(spLabelGT == i);
    vcc = unique(temp(temp > 0));
    %temp(temp > 0) = temp(temp > 0) + i*n_labs;
    for k = 1:numel(vcc)
      temp(temp == vcc(k)) = clabs(cci);
      cci = cci + 1;
    end
    connectCompGT = connectCompGT + temp;
end

pr = compare_segmentations(outImg, connectCompGT);
