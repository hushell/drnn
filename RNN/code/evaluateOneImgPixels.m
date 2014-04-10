function [correctPixels totalPixels] = evaluateOneImgPixels(imgData,catOut)

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
