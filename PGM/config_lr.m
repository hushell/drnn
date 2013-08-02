%%% --- configuration for logistic regression --- %%%
if ~exist('rmposfeat','var'),
    rmposfeat = 0;
end
if ~exist('verbose','var'),
    verbose = 0;
end

% lr
if ~exist('lrl2reg','var'),
    %lrl2reg = 0.003;
    lrl2reg = 0.01;
end

w_lr.params.rmposfeat = rmposfeat;
w_lr.params.l2reg = lrl2reg;
