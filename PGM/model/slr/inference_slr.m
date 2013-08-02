function mu = inference_slr(w_slr, feat, proj, nlabel)

% node potential
mu_node = zeros(nlabel,size(feat,1));
for i = 1:size(w_slr.nodeWeights,2),
    mu_node(i,:) = sum((squeeze(w_slr.nodeWeights(:,i,:))*proj).*feat',1);
end

% spatial logistic regression
mu_max = max(mu_node,[],1);
mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu_node,mu_max)),sum(exp(bsxfun(@minus,mu_node,mu_max)),1));

return;