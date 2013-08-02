function w_crf = crf_train(w_crf, Y, X, nlabel, l2reg_node, l2reg_edge)

% define the dimension of node and edge features
dimNodeFeat = size(X(1).nodeFeatures,1);
idx = find(X(1).adjmat > 0);
dimEdgeFeat = size(X(1).edgeFeatures{idx(1)},1);


% initialize crf weights
if ~exist('w_crf','var') || isempty(w_crf),
    w_crf.nodeWeights = 0.1*randn(dimNodeFeat,nlabel);
    w_crf.edgeWeights = 0.1*randn(dimEdgeFeat,nlabel,nlabel);
else
    if ~isstruct(w_crf),
        w_crf = struct;
    end
    if ~isfield(w_crf,'nodeWeights'),
        w_crf.nodeWeights = 0.1*randn(dimNodeFeat,nlabel);
    end
    if ~isfield(w_crf,'edgeWeights'),
        w_crf.edgeWeights = zeros(dimEdgeFeat,nlabel,nlabel);
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
gn_pos = zeros(size(w_crf.nodeWeights));
ge_pos = zeros(size(w_crf.edgeWeights));
k = 0;
for i = 1:N,
    Xc = X(i);
    Yc = Y{i};
    if size(Yc,1) == 1 || size(Yc,2) == 1,
        Yc = multi_output(Yc,nlabel);
    end
    % gradient
    [~, gnode_cur, gedge_cur] = cost_crf_sub(Xc, Yc, w_crf, 0, 0);
    
    gn_pos = gn_pos + gnode_cur;
    ge_pos = ge_pos + gedge_cur;
    k = k + Xc.numNodes;
end
gn_pos = gn_pos/k;
ge_pos = ge_pos/k;

% optimize edgeWeights first
fprintf('CRF: edge optimization only\n');
theta = w_crf.edgeWeights(:);
opttheta = minFunc(@(p) cost_crf_mf(p, w_crf, X, Y, gn_pos, ge_pos, l2reg_node, l2reg_edge, nlabel, dimNodeFeat, dimEdgeFeat), theta, options);
w_crf.edgeWeights = reshape(opttheta,dimEdgeFeat,nlabel,nlabel);

% jointly optimize
fprintf('CRF: joint node and edge optimization\n');
theta = [w_crf.nodeWeights(:) ; w_crf.edgeWeights(:)];
opttheta = minFunc(@(p) cost_crf_mf(p, w_crf, X, Y, gn_pos, ge_pos, l2reg_node, l2reg_edge, nlabel, dimNodeFeat, dimEdgeFeat), theta, options);
clear w_crf;
w_crf = unroll_pars_crf(opttheta, nlabel, dimNodeFeat, dimEdgeFeat);

return;
