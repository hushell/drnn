function [c, gn, ge] = cost_crf_sub(X, Y, w_crf, opt_negative, opt_node_fixed, opt_edge_fixed, opt_compute_cost, opt_fast)
threshold = 1e-10;

if ~exist('opt_node_fixed','var'),
    opt_node_fixed = 0;
end

if ~exist('opt_edge_fixed','var'),
    opt_edge_fixed = 0;
end

if ~exist('opt_negative','var'),
    opt_negative = 0;
end

if ~exist('opt_compute_cost','var'),
    opt_compute_cost = 1;
end

if ~exist('opt_fast','var'),
    opt_fast = 1;
end


%%% --- cost (log likelihood) --- %%%
if opt_compute_cost,
    % node
    c = sum(sum(Y.*(w_crf.nodeWeights'*X.nodeFeatures)));
    
    % edge
    [xi, xj] = find(X.adjmat > 0);
    if ~isfield(X,'edgeFeaturesrs'),
        [xi, xj] = find(X.adjmat > 0);
        edgeFeaturesrs = zeros(size(w_crf.edgeWeights,1),1,1,length(xi));
        for j = 1:length(xi),
            edgeFeaturesrs(:,:,:,j) = X.edgeFeatures{xi(j), xj(j)};
        end
        X = setfield(X,'edgeFeaturesrs',edgeFeaturesrs);
    end
    edgePot = squeeze(sum(bsxfun(@times,w_crf.edgeWeights,X.edgeFeaturesrs),1));
    c = c + sum(sum(bsxfun(@times,Y(:,xi),squeeze(sum(bsxfun(@times,edgePot,permute(Y(:,xj),[3 1 2])),2)))));
        
    % mean-field entropy
    if opt_negative,
        Y(Y<threshold) = 1;
        c = c - sum(Y(:).*log(Y(:)));
    end
else
    c = [];
end

%%% --- gradient --- %%%
if nargout == 3,
    if ~opt_node_fixed,
        % node
        gn = X.nodeFeatures*Y';
    else
        gn = [];
    end
    
    if ~opt_edge_fixed,
        % edge
        [xi, xj] = find(X.adjmat > 0);
        if opt_fast,
            if ~isfield(X,'edgeFeaturesrs'),
                edgeFeaturesrs = zeros(size(w_crf.edgeWeights,1),1,1,length(xi));
                for j = 1:length(xi),
                    edgeFeaturesrs(:,:,:,j) = X.edgeFeatures{xi(j), xj(j)};
                end
                X = setfield(X,'edgeFeaturesrs',edgeFeaturesrs);
            end
            ge = 0.5*sum(bsxfun(@times,bsxfun(@times,X.edgeFeaturesrs,permute(Y(:,xi),[3 1 4 2])),permute(Y(:,xj),[3 4 1 2])),4);
            ge = ge + 0.5*sum(bsxfun(@times,bsxfun(@times,X.edgeFeaturesrs,permute(Y(:,xj),[3 1 4 2])),permute(Y(:,xi),[3 4 1 2])),4);
        else
            ge = zeros(size(w_crf.edgeWeights));
            for i = 1:length(xi),
                ge = ge + 0.5*bsxfun(@times,X.edgeFeatures{xi(i), xj(i)},permute(Y(:,xi(i))*Y(:,xj(i))',[3 1 2]));
                ge = ge + 0.5*bsxfun(@times,X.edgeFeatures{xi(i), xj(i)},permute(Y(:,xj(i))*Y(:,xi(i))',[3 1 2]));
            end
        end
    else
        ge = [];
    end
end

return;