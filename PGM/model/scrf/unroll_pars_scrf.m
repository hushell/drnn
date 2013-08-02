function w_scrf = unroll_pars_scrf(theta, dim, nlabel, dimNodeFeat, dimEdgeFeat)

w_scrf = struct;
w_scrf.nodeWeights = double(reshape(theta(1:dimNodeFeat*nlabel*dim^2),dimNodeFeat,nlabel,dim^2));
w_scrf.edgeWeights = double(reshape(theta(dimNodeFeat*nlabel*dim^2+1:end),dimEdgeFeat,nlabel,nlabel));
% w_scrf.edgeWeights = double(reshape(theta,dimEdgeFeat,nlabel,nlabel));

return