function [subtree_ids, labelsUnder] = subtree_check_purity(...
    numLeafsUnder,leafsUnder,segLabels,numTotalNodes,numLeafNodes,purity,bg)
% purity: percentage of pure labels, used for thresholding subtrees
% bg: label index of bg, which will not be checked

subtree_ids = zeros(1,numTotalNodes-numLeafNodes);
labelsUnder = {};

cnt = 1;
for n = numLeafNodes+1:numTotalNodes
    labels = segLabels(leafsUnder{n});
    
    majority = mode(labels);
    if majority == bg
        continue;
    end
    
    if sum(labels == majority) / numel(labels) >= purity
        subtree_ids(n-numLeafNodes) = 1;
        labelsUnder{cnt} = labels;
        cnt = cnt + 1;
    end
end

subtree_ids = find(subtree_ids == 1) + numLeafNodes;
assert(length(subtree_ids) == length(labelsUnder));