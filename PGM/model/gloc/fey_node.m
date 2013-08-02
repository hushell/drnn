% nodeFeat      : numDim x numNodes
% nodeWeights   : numDim x numLabel x numNodes_crf
% proj          : numNodes_crf x numNodes
% prodFWP       : numLabel x numNodes
% mu            : numLabel x numNodes
% 
function [fey, prodFWP] = fey_node(mu,prodFWP,feat,w_scrf,proj_crf,params)

if ~isfield(feat,'nodeFeaturesrs'),
    feat.nodeFeaturesrs = permute(feat.nodeFeatures,[1 3 2]);
end

if ~isfield(w_scrf,'nodeWeightsrs'),
    w_scrf.nodeWeightsrs = reshape(w_scrf.nodeWeights,size(w_scrf.nodeWeights,1)*size(w_scrf.nodeWeights,2),size(w_scrf.nodeWeights,3));
end

if isempty(prodFWP),
    prodFWP = squeeze(sum(bsxfun(@times,feat.nodeFeaturesrs,reshape(w_scrf.nodeWeightsrs*proj_crf,params.numDim,params.numLabel,feat.numNodes)),1));
end

fey = -sum(prodFWP(:).*mu(:));

return;