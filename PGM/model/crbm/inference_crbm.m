function [mu, X] = inference_crbm(X, w_rbm, w_slr, params, proj_lr, proj_rbm, gamma)
if ~exist('gamma','var'),
    gamma = 1;
end

featSize = X.numNodes;

if isfield(X,'nodeFeatures'),
    X.nodeFeaturesrs = permute(X.nodeFeatures,[1 3 2]);
end

% node potential
mu_node = squeeze(sum(bsxfun(@times,X.nodeFeaturesrs,reshape(w_slr.nodeWeightsrs*proj_lr,params.numDim,params.numLabel,featSize)),1));

% initialize with lr
mu_max = max(mu_node,[],1);
mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu_node,mu_max)),sum(exp(bsxfun(@minus,mu_node,mu_max)),1));

check = 1;
iter = 0;
mu_old = mu;
while check,
    % hidden unit inference
    label_proj = proj_rbm*mu';
    ey = w_rbm.hidbiases + w_rbm.vishidrs'*label_proj(:);
    hidprob = sigmoid(ey);
    
    % energy from rbm
    mu_rbm = reshape(gamma*w_rbm.vishidrs*hidprob + w_rbm.visbiasrs,params.numNodes_rbm,params.numLabel);
    
    % aggregate energy from node and rbm
    mu_ey = mu_node + mu_rbm'*proj_rbm;
    mu_max = max(mu_ey,[],1);
    mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu_ey,mu_max)),sum(exp(bsxfun(@minus,mu_ey,mu_max)),1));
    
    err = norm(mu(:) - mu_old(:));
    if err < 1e-4 || iter > 100,
        check = 0;
    else
        mu_old = mu;
        iter = iter + 1;
    end
end

return;