function [forest, cut_if] = tree_cut_sample(imgTreeTop, p_connect)

numTotalNodes = size(imgTreeTop.kids,1);

if length(p_connect) == 1
  p_connect = repmat(p_connect,numTotalNodes,1);
end

assert(length(p_connect) == numTotalNodes);

% generate Bernulli R.V. from uniform distribution
% if unirand <= p_connect, cut_if = 0, i.e. connection = 1
% if unirand > p_connect, cut_if = 1
unirand = rand(numTotalNodes,1);
cut_if = unirand > p_connect;
cut_if(end) = 1; % always cut above root node

% get forest
forest = zeros(numTotalNodes,1); % indicate leafs belong to which tree 

% find the lowest cuts
for i = 1:numTotalNodes
    j = i;
    while 0 ~= j % parent(top) = 0
        if cut_if(j) == 1 
            forest(i) = j;
            break
        end
        j = imgTreeTop.pp(j); % go to parent
    end
end
