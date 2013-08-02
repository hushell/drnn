%%% --- configuration for CRF --- %%%
if ~exist('rmposfeat','var'),
    rmposfeat = 0;
end
if ~exist('verbose','var'),
    verbose = 0;
end

% crf
if ~exist('lrl2reg','var'),
    lrl2reg = 0.003;
end
if ~exist('l2reg_node','var'),
    l2reg_node = lrl2reg;
end
if ~exist('l2reg_edge','var'),
    l2reg_edge = 0.0001;
end

w_crf.params.rmposfeat = rmposfeat;
w_crf.params.l2reg_node = l2reg_node;
w_crf.params.l2reg_edge = l2reg_edge;
