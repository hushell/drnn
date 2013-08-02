function [fey, prodFWP, prodFW, prodPC, prodPW] = fey_gloc(mu, feat, w_scrf, w_rbm, params, proj_crf, proj_rbm, prodFWP, prodFW, prodPC, prodPW)

fey = 0;

% node
[fey_n, prodFWP] = fey_node(mu,prodFWP,feat,w_scrf,proj_crf,params);
fey = fey + fey_n;

% edge
[fey_e, prodFW] = fey_edge(mu,prodFW,feat,w_scrf,params);
fey = fey + fey_e;

% rbm
[fey_r, prodPC, prodPW] = fey_rbm(mu,prodPC,prodPW,w_rbm,proj_rbm,params);
fey = fey + fey_r;

return;