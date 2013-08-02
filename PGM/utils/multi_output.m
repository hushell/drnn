function y_mult = multi_output(y,nLabels)
%%%
% convert labels into 1-of-K representation
%
%   y       : 1       x batchsize
%   y_mult  : nLabels x batchsize
n = length(y);
y = double(y);
y_mult = sparse(1:n,y,1,n,nLabels,n);
y_mult = full(y_mult');
return;
