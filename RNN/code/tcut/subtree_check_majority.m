function [subtree_id, labelsUnder, purity] = subtree_check_majority(...
    numLeafsUnder,leafsUnder,segLabels,numTotalNodes,numLeafNodes,majority)
% output all subtrees that majority label is the query one

subtree_ids = zeros(1,numTotalNodes-numLeafNodes);
labelsUnder = {};

cnt = 1;
for n = numLeafNodes+1:numTotalNodes
    labels = segLabels(leafsUnder{n});

    if ismember(majority,labels)
        subtree_ids(n-numLeafNodes) = 1;
        labelsUnder{cnt} = labels;
        cnt = cnt + 1;
    end
end

subtree_ids = find(subtree_ids == 1) + numLeafNodes;
assert(length(subtree_ids) == length(labelsUnder));

if isempty(labelsUnder)
    subtree_id = -1;
    labelsUnder = {};
    purity = -1;
    return;
end

part_sp = find(segLabels == majority);
max_leafs = numLeafNodes+1;
subtree_id = 1;
for i = 1:length(subtree_ids)
    lia = ismember(part_sp, leafsUnder{subtree_ids(i)});
    if sum(lia) == length(part_sp)
        if numLeafsUnder(subtree_ids(i)) < max_leafs
            max_leafs = numLeafsUnder(subtree_ids(i));
            subtree_id = i;
        end
    end
end

labelsUnder = labelsUnder{subtree_id};
subtree_id = subtree_ids(subtree_id);
purity = sum(labelsUnder == majority) / numel(labelsUnder);