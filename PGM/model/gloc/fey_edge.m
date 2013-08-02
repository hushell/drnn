% prodFW        : (label*numNodes) x (label*numNodes)
% feat          : features
% w_scrf        : spatial crf weights
% mu            : numLabel x numNodes
% 
function [fey, prodFW] = fey_edge(mu,prodFW,feat,w_scrf,params)

if ~isfield(feat,'edgeFeaturesrs'),
    [xi, xj] = find(feat.adjmat > 0);
    edgeFeaturesrs = zeros(size(w_scrf.edgeWeights,1),1,1,length(xi));
    for fi = 1:length(xi),
        edgeFeaturesrs(:,:,:,fi) = feat.edgeFeatures{xi(fi), xj(fi)};
    end
    feat.edgeFeaturesrs = edgeFeaturesrs;
end

if isempty(prodFW),
    % edge feature
    [xi, xj] = find(feat.adjmat > 0);
    prodFW = zeros(params.numLabel,params.numLabel,feat.numNodes,feat.numNodes);
    for ei = 1:feat.numNodes,
        prodFW(:,:,ei,xj(xi == ei)) = reshape(sum(bsxfun(@times,feat.edgeFeaturesrs(:,:,:,xi == ei),w_scrf.edgeWeights),1),params.numLabel,params.numLabel,1,sum(xi == ei));
        prodFW(:,:,ei,xi(xj == ei)) = reshape(sum(bsxfun(@times,feat.edgeFeaturesrs(:,:,:,xj == ei),w_scrf.edgeWeights),1),params.numLabel,params.numLabel,1,sum(xj == ei));
    end
    prodFW = permute(prodFW,[1 3 2 4]);
    prodFW = reshape(prodFW,size(prodFW,1)*size(prodFW,2),size(prodFW,3)*size(prodFW,4));
end

fey = -mu(:)'*prodFW*mu(:);

return