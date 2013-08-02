function [c, gn, ge] = cost_spatial_crf_sub(X, Y, w_scrf, opt_negative, opt_node_fixed, opt_edge_fixed, opt_compute_cost, opt_fast)
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

proj = X.mapping_sp';

[dimNodeFeat,nlabel,~] = size(w_scrf.nodeWeights);
featSize = X.numNodes;
if ~isfield(w_scrf,'nodeWeightsrs'),
    w_scrf.nodeWeightsrs = reshape(w_scrf.nodeWeights,size(w_scrf.nodeWeights,1)*size(w_scrf.nodeWeights,2),size(w_scrf.nodeWeights,3));
end
nodeFeaturesrs = permute(X.nodeFeatures,[1 3 2]);

%%% --- cost (log likelihood) --- %%%
if opt_compute_cost,
    if opt_fast,
        % node
        c = sum(sum(Y.*squeeze(sum(bsxfun(@times,nodeFeaturesrs,reshape(w_scrf.nodeWeightsrs*proj,dimNodeFeat,nlabel,featSize)),1))));
        
        % edge
        [xi, xj] = find(X.adjmat > 0);
        if ~isfield(X,'edgeFeaturesrs'),
            [xi, xj] = find(X.adjmat > 0);
            edgeFeaturesrs = zeros(size(w_scrf.edgeWeights,1),1,1,length(xi));
            for j = 1:length(xi),
                edgeFeaturesrs(:,:,:,j) = X.edgeFeatures{xi(j), xj(j)};
            end
            X = setfield(X,'edgeFeaturesrs',edgeFeaturesrs);
        end
        edgePot = squeeze(sum(bsxfun(@times,w_scrf.edgeWeights,X.edgeFeaturesrs),1));
        c = c + sum(sum(bsxfun(@times,Y(:,xi),squeeze(sum(bsxfun(@times,edgePot,permute(Y(:,xj),[3 1 2])),2)))));
    else
        c = 0;
        
        % nodePotential
        nodePotential = zeros(nlabel, featSize);
        for i = 1:featSize,
            for l = 1:nlabel,
                nodePotential(l,i) = X.nodeFeatures(:,i)'*(squeeze(w_scrf.nodeWeights(:,l,:))*proj(:,i));
            end
        end
        
        % edgePotential
        [edgeIdx1, edgeIdx2] = find(X.adjmat > 0);
        nEdges = length(edgeIdx1);
        edgePotential = zeros(nlabel, nlabel, nEdges);
        for i = 1:nEdges,
            for l1 = 1:nlabel,
                for l2 = 1:nlabel,
                    edgePotential(l1,l2,i) = sum(w_scrf.edgeWeights(:,l1,l2).*X.edgeFeatures{edgeIdx1(i),edgeIdx2(i)});
                end
            end
        end
        
        % node
        for i = 1:featSize,
            for l = 1:nlabel,
                c = c + Y(l,i)*nodePotential(l,i);
            end
        end
        
        % edge
        for i = 1:nEdges
            n1 = edgeIdx1(i);
            n2 = edgeIdx2(i);
            for l1 = 1:nlabel,
                for l2 = 1:nlabel,
                    c = c + Y(l1,n1)*Y(l2,n2)*edgePotential(l1,l2,i);
                end
            end
        end
    end
    
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
        gn = reshape(reshape(bsxfun(@times,nodeFeaturesrs,permute(Y,[3 1 2])),dimNodeFeat*nlabel,featSize)*proj',size(w_scrf.nodeWeights));
    else
        gn = [];
    end
    
    if ~opt_edge_fixed,
        % edge
        [xi, xj] = find(X.adjmat > 0);
        if opt_fast,
            if ~isfield(X,'edgeFeaturesrs'),
                edgeFeaturesrs = zeros(size(w_scrf.edgeWeights,1),1,1,length(xi));
                for j = 1:length(xi),
                    edgeFeaturesrs(:,:,:,j) = X.edgeFeatures{xi(j), xj(j)};
                end
                X = setfield(X,'edgeFeaturesrs',edgeFeaturesrs);
            end
            ge = 0.5*sum(bsxfun(@times,bsxfun(@times,X.edgeFeaturesrs,permute(Y(:,xi),[3 1 4 2])),permute(Y(:,xj),[3 4 1 2])),4);
            ge = ge + 0.5*sum(bsxfun(@times,bsxfun(@times,X.edgeFeaturesrs,permute(Y(:,xj),[3 1 4 2])),permute(Y(:,xi),[3 4 1 2])),4);
        else
            ge = zeros(size(w_scrf.edgeWeights));
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