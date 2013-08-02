function [mu, X] = inference_gloc(X, w_gloc, params, proj_crf, proj_rbm)

featSize = X.numNodes;

if ~isfield(X,'edgeFeaturesrs'),
    [xi, xj] = find(X.adjmat > 0);
    edgeFeaturesrs = zeros(size(w_gloc.edgeWeights,1),1,1,length(xi));
    for fi = 1:length(xi),
        edgeFeaturesrs(:,:,:,fi) = X.edgeFeatures{xi(fi), xj(fi)};
    end
    X.edgeFeaturesrs = edgeFeaturesrs;
end

if isfield(X,'nodeFeatures'),
    X.nodeFeaturesrs = permute(X.nodeFeatures,[1 3 2]);
end

% edge feature
[xi, xj] = find(X.adjmat > 0);
edgeFeat = zeros(params.numLabel,params.numLabel,X.numNodes,X.numNodes);
for ei = 1:featSize,
    edgeFeat(:,:,ei,xj(xi == ei)) = reshape(sum(bsxfun(@times,X.edgeFeaturesrs(:,:,:,xi == ei),w_gloc.edgeWeights),1),params.numLabel,params.numLabel,1,sum(xi == ei));
    edgeFeat(:,:,ei,xi(xj == ei)) = reshape(sum(bsxfun(@times,X.edgeFeaturesrs(:,:,:,xj == ei),w_gloc.edgeWeights),1),params.numLabel,params.numLabel,1,sum(xj == ei));
end
edgeFeat = permute(edgeFeat,[1 3 2 4]);
edgeFeat = reshape(edgeFeat,size(edgeFeat,1)*size(edgeFeat,2),size(edgeFeat,3)*size(edgeFeat,4));

% node potential
mu_node = squeeze(sum(bsxfun(@times,X.nodeFeaturesrs,reshape(w_gloc.nodeWeightsrs*proj_crf,params.numDim,params.numLabel,featSize)),1));


% initialize with lr
mu_max = max(mu_node,[],1);
mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu_node,mu_max)),sum(exp(bsxfun(@minus,mu_node,mu_max)),1));
% stop when converge, but use the sample with the lowest energy
fey = inf;
prodFWP = [];
prodFW = [];
prodPC = [];
prodPW = [];
check = 1;
iter = 0;

mu_old_outer = mu;
% mean-field iteration
while check,
    % hidden unit inference
    label_proj = proj_rbm*mu';
    ey = w_gloc.hidbiases + w_gloc.vishidrs'*label_proj(:);
    hidprob = sigmoid(ey);
    
    % energy from rbm
    mu_rbm = reshape(w_gloc.vishidrs*hidprob + w_gloc.visbiasrs,params.numNodes_rbm,params.numLabel);
    
    % energy from node and rbm
    mu_ey = mu_node + mu_rbm'*proj_rbm;
    
    % energy from edge
    mu = mu_ey + reshape(edgeFeat*mu(:),params.numLabel,size(edgeFeat,1)/params.numLabel);
    
    mu_max = max(mu,[],1);
    mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu,mu_max)),sum(exp(bsxfun(@minus,mu,mu_max)),1));
    
    [fey_cur, prodFWP, prodFW, prodPC, prodPW] = fey_gloc(mu, X, w_gloc, w_gloc, params, proj_crf, proj_rbm, prodFWP, prodFW, prodPC, prodPW);
    if fey_cur < fey,
        mu_lowest_ey = mu;
        fey = fey_cur;
    end
    
    err = norm(mu(:) - mu_old_outer(:));
    if err < 1e-4 || iter > 200,
        check = 0;
    else
        mu_old_outer = mu;
        iter = iter + 1;
    end
end
mu = mu_lowest_ey;

return;
