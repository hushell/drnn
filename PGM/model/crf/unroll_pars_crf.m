function w_crf = unroll_pars_crf(theta, nlabel, dimNodeFeat, dimEdgeFeat)

w_crf = struct;
w_crf.nodeWeights = double(reshape(theta(1:dimNodeFeat*nlabel),dimNodeFeat,nlabel));
w_crf.edgeWeights = double(reshape(theta(dimNodeFeat*nlabel+1:end),dimEdgeFeat,nlabel,nlabel));
% w_crf.edgeWeights = double(reshape(theta,dimEdgeFeat,nlabel,nlabel));

return