function [f, grad] = cost_lr(theta,X,Y,num_in,num_out,lambda)

W = reshape(theta(1:num_in*num_out),num_in,num_out);

Wgrad = zeros(size(W));

% forward propagation
Yr = inference_lr(X,W);

% loss function
[f, out, dout] = copmuteLoss(Yr, Y);

% add up regularization penalties
f = f + 0.5*lambda*sum(W(:).^2);

da = out.*dout - out.*(repmat(sum(out.*dout),size(out,1),1));
Wgrad = Wgrad + X*da';
Wgrad = Wgrad + lambda*W;

grad = Wgrad(:);

return;

function [f, out, dout] = copmuteLoss(Yr, Y)
%%% compute loss function
f = - full(mean(sum(Y.*log(Yr))));
out = Yr;
dout = -Y.*(1./out)/size(Yr,2);
return;
