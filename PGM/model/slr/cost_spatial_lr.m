function [f, grad] = cost_spatial_lr(theta,feat,Y,proj,num_in,num_out,lambda)

batchSize = length(feat);
W = reshape(theta,length(theta)/num_in/num_out,num_in,num_out);

Wgrad = zeros(size(W));

% forward propagation (prediction)
Wrs = reshape(W,size(W,1),size(W,2)*size(W,3));
Yr = zeros(size(Y));
k = 0;
for i = 1:batchSize,
    curPs = size(proj{i},2);
    Yr(:,k+1:k+curPs) = squeeze(sum(bsxfun(@times,double(feat{i}),reshape(double(proj{i})'*Wrs,curPs,num_in,num_out)),2))';
    k = k + curPs;
end
Yr = exp(bsxfun(@minus, Yr, max(Yr, [], 1)));
Yr = bsxfun(@rdivide, Yr, sum(Yr,1));

% loss function
[f, out, dout] = copmuteLoss(Yr, Y);

% add up regularization penalties
f = f + 0.5*lambda*sum(W(:).^2);

da = out.*dout - out.*(repmat(sum(out.*dout),size(out,1),1));

da = permute(da',[1 3 2]);
k = 0;
for i = 1:batchSize,
    curPs = size(proj{i},2);
    Wgrad = Wgrad + reshape(double(proj{i})*reshape(bsxfun(@times,double(feat{i}),da(k+1:k+curPs,1,:)),curPs,num_in*num_out),size(W));
    k = k + curPs;
end
Wgrad = Wgrad + lambda*W;

grad = Wgrad(:);

return;

function [f, out, dout] = copmuteLoss(Yr, Y)
%%% compute loss function
f = - full(mean(sum(Y.*log(Yr))));
out = Yr;
dout = -Y.*(1./out)/size(Yr,2);
return;
