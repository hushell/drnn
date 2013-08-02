function [c, g] = cost_crf_mf(theta, w_crf, X, Y, gnode_pos, gedge_pos, l2reg_node, l2reg_edge, nlabel, dimNodeFeat, dimEdgeFeat)

try
    w_crf = unroll_pars_crf(theta, nlabel, dimNodeFeat, dimEdgeFeat);
    opt_node_fixed = 0;
    opt_edge_fixed = 0;
catch
    w_crf.edgeWeights = reshape(theta,size(w_crf.edgeWeights));
    opt_node_fixed = 1;
    opt_edge_fixed = 0;
end

if ~isempty(gnode_pos) && ~isempty(gedge_pos),
    opt_compute_pos = 0;
else
    opt_compute_pos = 1;
end

% initialize function outputs
c = 0;
gnode = zeros(size(w_crf.nodeWeights));
gedge = zeros(size(w_crf.edgeWeights));

if ~opt_node_fixed,
    if opt_compute_pos,
        gnode_pos = zeros(size(w_crf.nodeWeights));
    end
    gnode_neg = zeros(size(w_crf.nodeWeights));
else
    gnode_neg = [];
end

if ~opt_edge_fixed,
    if opt_compute_pos,
        gedge_pos = zeros(size(w_crf.edgeWeights));
    end
    gedge_neg = zeros(size(w_crf.edgeWeights));
else
    gedge_neg = [];
end

N = length(X);

%%% --- positive phase --- %%%
c_pos = 0;
if opt_compute_pos,
    k = 0;
    for i = 1:N,
        Xc = X(i);
        Yc = Y{i};
        if size(Yc,1) == 1 || size(Yc,2) == 1,
            Yc = multi_output(Yc,nlabel);
        end
        % gradient
        [c_cur, gnode_cur, gedge_cur] = cost_crf_sub(Xc, Yc, w_crf, 0, opt_node_fixed, opt_edge_fixed);
        
        c_pos = c_pos + c_cur;
        gnode_pos = gnode_pos + gnode_cur;
        gedge_pos = gedge_pos + gedge_cur;
        k = k + Xc.numNodes;
    end
    c_pos = c_pos/k;
    gnode_pos = gnode_pos/k;
    gedge_pos = gedge_pos/k;
else
    k = 0;
    for i = 1:N,
        Xc = X(i);
        Yc = Y{i};
        if size(Yc,1) == 1 || size(Yc,2) == 1,
            Yc = multi_output(Yc,nlabel);
        end
        % gradient
        c_cur = cost_crf_sub(Xc, Yc, w_crf, 0);
        
        c_pos = c_pos + c_cur;
        k = k + Xc.numNodes;
    end
    c_pos = c_pos/k;
end


%%% --- negative phase --- %%%
c_neg = 0;
k = 0;
for i = 1:N,
    Xc = X(i);
    Yc = inference_crf(w_crf, Xc, nlabel, 0);
    
    % gradient
    [c_cur, gnode_cur, gedge_cur] = cost_crf_sub(Xc, Yc, w_crf, 1, opt_node_fixed, opt_edge_fixed);
    
    c_neg = c_neg + c_cur;
    gnode_neg = gnode_neg + gnode_cur;
    gedge_neg = gedge_neg + gedge_cur;
    k = k + Xc.numNodes;
end
c_neg = c_neg/k;
gnode_neg = gnode_neg/k;
gedge_neg = gedge_neg/k;

% positive phase
c = c + c_pos;
if ~opt_node_fixed,
    gnode = gnode + gnode_pos;
end
if ~opt_edge_fixed,
    gedge = gedge + gedge_pos;
end

% negative phase
c = c - c_neg;
if ~opt_node_fixed,
    gnode = gnode - gnode_neg;
end
if ~opt_edge_fixed,
    gedge = gedge - gedge_neg;
end

% invert the sign
c = -c;
gnode = -gnode;
gedge = -gedge;

%%% --- regularziation --- %%%
% node
if ~opt_node_fixed,
    c = c + 0.5*l2reg_node*sum(w_crf.nodeWeights(:).^2);
    gnode = gnode + l2reg_node*w_crf.nodeWeights;
end

% edge
if ~opt_edge_fixed,
    c = c + 0.5*l2reg_edge*sum(w_crf.edgeWeights(:).^2);
    gedge = gedge + l2reg_edge*w_crf.edgeWeights;
end

g = [];
if ~opt_node_fixed,
    g = [g ; gnode(:)];
end
if ~opt_edge_fixed,
    g = [g ; gedge(:)];
end

fprintf('.');
return;