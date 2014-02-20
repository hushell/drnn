function [y,I] = fullmax(X)
%

% new version of max, that for X as a matrix, returns in I the indices
% of all maximizing entries per column not just the first one
% is X is a row vector, retunrs the scalar maximizing value and  
% the indices of all maximizing entries
%
% input X: matrix
% output y: row vector (max per col)
% output I: matrix same size as X with 1's for maximizer in each col
%
% input X: row vector
% output y: scalar (max value of the row)
% output I: matrix same size as X with 1's for maximizer in each col

  
y = max(X); % original max function
if (size(X,1)==1)  % row vector
  Y = y*ones(size(X));
else  
  Y = repmat(y,size(X,1),1);
end
%I = (Y==X);
I = find(Y==X);



