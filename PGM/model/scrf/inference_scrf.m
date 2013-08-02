function [mu, flag] = inference_scrf(w_scrf, X, nlabel, nMF)
% if flag = 0, mean-field didn't converge, no update
% if flag = 1, mean-field converged, update

if ~exist('nMF','var') || nMF == 0,
    nMF = 200;
end

proj = X.mapping_sp';

% node potential
mu_node = zeros(nlabel,X.numNodes);
for ni = 1:size(w_scrf.nodeWeights,2),
    mu_node(ni,:) = sum(X.nodeFeatures.*(squeeze(w_scrf.nodeWeights(:,ni,:))*proj),1);
end

% initialize with linear regression
mu_max = max(mu_node,[],1);
mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu_node,mu_max)),sum(exp(bsxfun(@minus,mu_node,mu_max)),1));

if ~isfield(X,'edgeFeaturesrs'),
    [xi, xj] = find(X.adjmat > 0);
    edgeFeaturesrs = zeros(size(w_scrf.edgeWeights,1),1,1,length(xi));
    for j = 1:length(xi),
        edgeFeaturesrs(:,:,:,j) = X.edgeFeatures{xi(j), xj(j)};
    end
    X.edgeFeaturesrs = edgeFeaturesrs;
end

% edge potential (preprocess)
[xi, xj] = find(X.adjmat > 0);
edgeFeat = zeros(nlabel,nlabel,X.numNodes,X.numNodes);
for ei = 1:X.numNodes,
    edgeFeat(:,:,ei,xj(xi == ei)) = reshape(sum(bsxfun(@times,X.edgeFeaturesrs(:,:,:,xi == ei),w_scrf.edgeWeights),1),nlabel,nlabel,1,sum(xi == ei));
    edgeFeat(:,:,ei,xi(xj == ei)) = reshape(sum(bsxfun(@times,X.edgeFeaturesrs(:,:,:,xj == ei),w_scrf.edgeWeights),1),nlabel,nlabel,1,sum(xj == ei));
end

edgeFeat = permute(edgeFeat,[1 3 2 4]);
edgeFeat = reshape(edgeFeat,size(edgeFeat,1)*size(edgeFeat,2),size(edgeFeat,3)*size(edgeFeat,4));

flag = 0;
% mean-field iteration (full block)
mu_old = mu;
for nmf = 1:nMF,
    % mu: nLabels x numFeat
    mu = mu_node + reshape(edgeFeat*mu(:),nlabel,size(edgeFeat,1)/nlabel);
    
    mu_max = max(mu,[],1);
    mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu,mu_max)),sum(exp(bsxfun(@minus,mu,mu_max)),1));
    err = norm(mu(:) - mu_old(:));
    if err < 1e-4,
        flag = 1;
        break;
    else
        mu_old = mu;
    end
end

return;
