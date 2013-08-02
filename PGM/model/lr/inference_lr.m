function Yr = inference_lr(X,W)
%%% softmax prediction
a = W'*X;
a = exp(bsxfun(@minus, a, max(a, [], 1)));
Yr = bsxfun(@rdivide, a, sum(a,1));
return;
