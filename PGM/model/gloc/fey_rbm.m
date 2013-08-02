% nodeFeat      : numDim x numNodes
% nodeWeights   : numDim x numLabel x numNodes_crf
% proj          : numNodes_crf x numNodes
% prodFWP       : numLabel x numNodes
% mu            : numLabel x numNodes
% 
function [fey, prodPC, prodPW] = fey_rbm(mu,prodPC,prodPW,w_rbm,proj_rbm,params)

mu_vec = mu(:);

if isempty(prodPC),
    prodPC = w_rbm.visbiases'*proj_rbm;
    prodPC = prodPC(:);
end
    
if ~isfield(w_rbm,'vishidperm')
    w_rbm.vishidperm = reshape(w_rbm.vishid,size(w_rbm.vishid,1),size(w_rbm.vishid,2)*size(w_rbm.vishid,3));
end

if isempty(prodPW),
    prodPW = w_rbm.vishidperm'*proj_rbm;
    prodPW = reshape(prodPW,params.numLabel,params.numHid,size(proj_rbm,2));
    prodPW = permute(prodPW,[1 3 2]);
    prodPW = reshape(prodPW,size(prodPW,1)*size(prodPW,2),size(prodPW,3));
end

fey = -prodPC'*mu_vec;
fey = fey - sum(logexp(prodPW'*mu_vec));

return;

function y = logexp(x)
idx = x > 0;
idx2 = -2*idx + 1;

y = idx.*x + log(1+exp(idx2.*x));

return;