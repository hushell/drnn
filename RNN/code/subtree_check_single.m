function [subtree_ids, labelsUnder, purities] = subtree_check_single(imgTreeTop,imgData,purity,bg)
% the hypothesis was the max pure subtree == part
% however, found pure subtrees may be too small, i.e. a subregion of part
% so we can check majority, specifically, find the min subtree contains 
% all the SPs of the part. The worst case is the whole parse tree
%
% this function has 2 purposes:
% 1) output all pure subtrees: purity in [0,1]
% 2) output all subtrees corresponding to parts: purity < 0
%
% bg: label index of bg, which will not be checked

if nargin < 4
    bg = 7;
end

numLeafNodes = size(imgData.adj,1);
numTotalNodes = size(imgTreeTop.kids,1);

numLeafsUnder = ones(numLeafNodes,1);
leafsUnder = cell(numLeafNodes,1);
for s = 1:numLeafNodes
    leafsUnder{s} = s;
end

for n = numLeafNodes+1:numTotalNodes
    kids = imgTreeTop.getKids(n);
    numLeafsUnder(n) = numLeafsUnder(kids(1))+numLeafsUnder(kids(2));
    leafsUnder{n} = [leafsUnder{kids(1)} leafsUnder{kids(2)}];
end

segLabels = imgData.segLabels;

if purity < 0
    subtree_ids = cell(1,max(segLabels));
    labelsUnder = cell(1,max(segLabels));
    purities = cell(1,max(segLabels));
    for mi = 1:max(segLabels)
        if mi == bg
            continue;
        end
        [subtree_ids{mi}, labelsUnder{mi}, purities{mi}] = subtree_check_majority(...
            numLeafsUnder,leafsUnder,segLabels,numTotalNodes,numLeafNodes,mi);
    end
else
    assert(purity >= 0 && purity <= 1);
    purities = [];
    [subtree_ids, labelsUnder] = subtree_check_purity(numLeafsUnder,...
        leafsUnder,segLabels,numTotalNodes,numLeafNodes,purity,bg);
end

