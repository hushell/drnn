% pretrain spatial conditional random fields on LFW part label dataset
% while fixing the node weights
% 
% we use (block) mean-field for inference and LBFGS for optimization.
% (we use LBFGS in minFunc by Mark Schmidt: 
% http://www.di.ens.fr/~mschmidt/Software/minFunc.html)
%
% input
%   w_scrf          : field - nodeWeights (pretrained from slr)
%   Y               : cell, numNodes x 1
%   X               : cell, field - nodeFeatures, edgeWeights, adjmat, numNodes
%   dim             : N (see reference)
%   nlabel          : number of part labels (e.g., 3 in LFW part label dataset)
%   l2reg_node      : weight decay for node weights
%   l2reg_edge      : weight decay for edge weights
% 
% output
%   w_scrf          : field - nodeWeights (pretrained from slr), edgeWeights
%
% written by Kihyuk Sohn, 2013/04/20
%
% reference:
% Augmenting CRFs with Boltzmann Machine Shape Priors for Image Labeling, CVPR, 2013.
%

function w_scrf = spatial_crf_train_edge_only(w_scrf, Y, X, dim, nlabel, l2reg_node, l2reg_edge)

% define the dimension of node and edge features
dimNodeFeat = size(X(1).nodeFeatures,1);
idx = find(X(1).adjmat > 0);
dimEdgeFeat = size(X(1).edgeFeatures{idx(1)},1);

% initialize spatial crf weights
if ~exist('w_scrf','var') || isempty(w_scrf),
    w_scrf.nodeWeights = 0.1*randn(dimNodeFeat,nlabel,dim^2);
    w_scrf.edgeWeights = 0.1*randn(dimEdgeFeat,nlabel,nlabel);
else
    if ~isstruct(w_scrf),
        w_scrf = struct;
    end
    if ~isfield(w_scrf,'nodeWeights'),
        w_scrf.nodeWeights = 0.1*randn(dimNodeFeat,nlabel,dim^2);
    end
    if ~isfield(w_scrf,'edgeWeights'),
        w_scrf.edgeWeights = zeros(dimEdgeFeat,nlabel,nlabel);
    end
end

options.maxIter = 400;
options.maxFunEvals = 400;

N = length(X);
%%% --- create X.edgeFeaturesrs --- %%%
for i = 1:N,
    Xc = X(i);
    [xi, xj] = find(Xc.adjmat > 0);
    edgeFeaturesrs = zeros(dimEdgeFeat,1,1,length(xi));
    for j = 1:length(xi),
        edgeFeaturesrs(:,:,:,j) = Xc.edgeFeatures{xi(j), xj(j)};
    end
    X(i).edgeFeaturesrs = edgeFeaturesrs;
end

%%% --- Positive gradients --- %%%
gn_pos = zeros(size(w_scrf.nodeWeights));
ge_pos = zeros(size(w_scrf.edgeWeights));
k = 0;
for i = 1:N,
    Xc = X(i);
    Yc = Y{i};
    if size(Yc,1) == 1 || size(Yc,2) == 1,
        Yc = multi_output(Yc,nlabel);
    end
    % gradient
    [~, gnode_cur, gedge_cur] = cost_spatial_crf_sub(Xc, Yc, w_scrf, 0, 0);
    
    gn_pos = gn_pos + gnode_cur;
    ge_pos = ge_pos + gedge_cur;
    k = k + Xc.numNodes;
end
gn_pos = gn_pos/k;
ge_pos = ge_pos/k;

% optimize edgeWeights first
fprintf('SCRF: edge optimization only\n');
theta = w_scrf.edgeWeights(:);
opttheta = minFunc(@(p) cost_spatial_crf_mf(p, w_scrf, X, Y, gn_pos, ge_pos, l2reg_node, l2reg_edge, dim, nlabel, dimNodeFeat, dimEdgeFeat), theta, options);
w_scrf.edgeWeights = reshape(opttheta,dimEdgeFeat,nlabel,nlabel);

return;
