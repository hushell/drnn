% Pretrain conditional restricted Boltzmann machines on LFW labeling dataset.
%
% we followed pretraining technique of top layer for
% deep Boltzmann machines (e.g., duplicate the hidden units
% during the training)
%
% input
%   tr_feat         : cell, field - nodeFeatures, numNodes
%   tr_proj_crf     : cell, {p_{sn}}
%   tr_proj_rbm     : cell, {p_{rs}}
%   tr_label        : cell, numNodes x 1
%   params          : hyperparameters
%   w_slr           : field - nodeWeights (pretrained from slr)
%
% output
%   w_rbm           : field - vishid, hidbiases, visbiases
%   w_slr           : field - nodeWeights
%   params          : hyperparameters
% 
%
% reference:
% Augmenting CRFs with Boltzmann Machine Shape Priors for Image Labeling, CVPR, 2013.
%

function [w_rbm, w_slr, params] = crbm_train(tr_feat, tr_proj_crf, tr_proj_rbm, tr_label, params, w_slr)

warning off all;
close all;
rng('shuffle');

fname_date = sprintf('%s_date%s',params.fname,datestr(now, 30));

initialmomentum  = 0.5;
finalmomentum    = 0.9;

% initialize weight
w_rbm.vishid = 0.1*randn(params.numNodes_rbm, params.numLabel-1, params.numHid);
w_rbm.visbiases = zeros(params.numNodes_rbm,params.numLabel-1);
w_rbm.hidbiases = zeros(params.numHid,1);
if size(w_rbm.vishid,2) < params.numLabel,
    w_rbm.vishid(:,size(w_rbm.vishid,2)+1:params.numLabel,:) = 0;
end
if size(w_rbm.visbiases,2) < params.numLabel,
    w_rbm.visbiases(:,size(w_rbm.visbiases,2)+1:params.numLabel) = 0;
end

if ~exist('w_slr','var') || isempty(w_slr),
    w_slr.nodeWeights = 0.1*randn(params.numDim,params.numLabel,params.numNodes_crf);
end

vishidinc = zeros(size(w_rbm.vishid));
vbiasinc = zeros(size(w_rbm.visbiases));
hbiasinc = zeros(size(w_rbm.hidbiases));
nodeWinc = zeros(size(w_slr.nodeWeights));

disp(params);

%%% --- learning --- %%%
mu_node = cell(params.batchSize,1);
label_mult = cell(params.batchSize,1);
neglabel = cell(params.batchSize,1);
label_proj = zeros(params.numNodes_rbm,params.numLabel,params.batchSize);

for j = 1:params.batchSize,
    label = tr_label{j};
    label_mult{j} = multi_output(label,params.numLabel)'; % numDim x numCh
end

fname = sprintf('%s_%04d',params.fname,0);
save_params(params,w_rbm,w_slr,0,fname);

for t = 1:params.maxepoch,
    % momentum update
    if t < params.momch,
        momentum = initialmomentum;
    else
        momentum = finalmomentum;
    end
    
    % # CD steps, learning rate
    if params.anneal,
        KCD = min(max(ceil(t/10),5),30);
        if t > params.maxepoch,
            KCD = max(KCD, params.KCD);
        end
        epsilon = params.epsilon/(1+params.epsdecay*KCD);
    else
        KCD = params.KCD;
        epsilon = params.epsilon/(1+params.epsdecay*t);
    end
    
    % reshape variables to speed up
    w_rbm.vishidrs = reshape(w_rbm.vishid,size(w_rbm.vishid,1)*size(w_rbm.vishid,2),size(w_rbm.vishid,3));
    w_rbm.visbiasrs  = reshape(w_rbm.visbiases,size(w_rbm.vishid,1)*size(w_rbm.vishid,2),1);
    w_slr.nodeWeightsrs = reshape(w_slr.nodeWeights,size(w_slr.nodeWeights,1)*size(w_slr.nodeWeights,2),size(w_slr.nodeWeights,3));
    
    tic;
    %%% ----- positive phase ----- %%%
    %%% hidden unit inference
    for j = 1:params.batchSize,
        proj_rbm = tr_proj_rbm{j};
        label_proj(:,:,j) = proj_rbm*label_mult{j};
    end
    ey = bsxfun(@plus,w_rbm.hidbiases,w_rbm.vishidrs'*reshape(label_proj,params.numNodes_rbm*params.numLabel,params.batchSize));
    poshidprob = sigmoid(ey);
    % twice the hidden units
    poshidstate1 = rand(size(poshidprob)) < poshidprob;
    poshidstate2 = rand(size(poshidprob)) < poshidprob;
    
    % positive gradient
    dvh_pos = 2*reshape(reshape(label_proj,params.numNodes_rbm*params.numLabel,params.batchSize)*poshidprob',size(w_rbm.vishid))/params.batchSize;
    dh_pos = 2*mean(poshidprob,2);
    dv_pos = mean(label_proj,3);
    
    
    %%% visible unit inference
    % energy from rbm
    mu_rbm = reshape(bsxfun(@plus,w_rbm.vishidrs*(poshidprob + poshidprob),w_rbm.visbiasrs),params.numNodes_rbm,params.numLabel,params.batchSize);
    
    recon_err_all = 0;
    for j = 1:params.batchSize,
        feat = tr_feat{j};
        proj_rbm = tr_proj_rbm{j};
        proj_crf = tr_proj_crf{j};
        
        if isfield(feat,'nodeFeatures'),
            feat.nodeFeaturesrs = permute(feat.nodeFeatures,[1 3 2]);
            feat = rmfield(feat,'nodeFeatures');
            tr_feat{j} = feat;
        end
        
        % node potential
        mu_node{j} = squeeze(sum(bsxfun(@times,feat.nodeFeaturesrs,reshape(w_slr.nodeWeightsrs*proj_crf,params.numDim,params.numLabel,feat.numNodes)),1));
        
        % aggregate energy from node and rbm
        mu_ey = mu_node{j} + mu_rbm(:,:,j)'*proj_rbm;
        mu_max = max(mu_ey,[],1);
        mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu_ey,mu_max)),sum(exp(bsxfun(@minus,mu_ey,mu_max)),1));
        neglabel{j} = mu';
        recon_err = norm(label_mult{j} - neglabel{j});
        recon_err_all = recon_err_all + recon_err;
    end
    
    %%% ----- negative phase ----- %%%
    neghidstate1 = zeros(size(poshidstate1));
    neghidstate2 = zeros(size(poshidstate2));
    for kcd = 1:KCD,
        %%% visible unit inference
        % energy from rbm
        mu_rbm = reshape(bsxfun(@plus,w_rbm.vishidrs*(neghidstate1+neghidstate2),w_rbm.visbiasrs),params.numNodes_rbm,params.numLabel,params.batchSize);
        
        for j = 1:params.batchSize,
            % aggregate energy from node and w_rbm_pre
            proj_rbm = tr_proj_rbm{j};
            mu_ey = mu_node{j} + mu_rbm(:,:,j)'*proj_rbm;
            
            mu_max = max(mu_ey,[],1);
            mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu_ey,mu_max)),sum(exp(bsxfun(@minus,mu_ey,mu_max)),1));
            neglabel{j} = mu';
            
            label_proj(:,:,j) = proj_rbm*neglabel{j};
        end
        
        %%% hidden unit inference
        ey = bsxfun(@plus,w_rbm.hidbiases,w_rbm.vishidrs'*reshape(label_proj,params.numNodes_rbm*params.numLabel,params.batchSize));
        neghidprob = sigmoid(ey);
        neghidstate1 = rand(size(neghidprob)) < neghidprob;
        neghidstate2 = rand(size(neghidprob)) < neghidprob;
    end
    
    % negative gradient
    dvh_neg = 2*reshape(reshape(label_proj,params.numNodes_rbm*params.numLabel,params.batchSize)*neghidprob',size(w_rbm.vishid))/params.batchSize;
    dh_neg = 2*mean(neghidprob,2);
    dv_neg = mean(label_proj,3);
    
    % gradients
    dvh_total = (dvh_pos - dvh_neg);
    dh_total = (dh_pos - dh_neg);
    dv_total = (dv_pos - dv_neg);
    dnW_total = zeros(size(w_slr.nodeWeights));
    
    if params.opt_nodeup,
        k = 0;
        for j = 1:params.batchSize,
            feat = tr_feat{j};
            proj_crf = tr_proj_crf{j};
            dnW_total = dnW_total + reshape(reshape(bsxfun(@times,feat.nodeFeaturesrs,permute(label_mult{j}-neglabel{j},[3 2 1])),params.numDim*params.numLabel,feat.numNodes)*proj_crf',params.numDim,params.numLabel,params.numNodes_crf);
            k = k + feat.numNodes;
        end
        dnW_total = dnW_total/k;
    end
    toc;
    
    vishidinc = momentum*vishidinc + epsilon*(dvh_total - params.l2reg*w_rbm.vishid);
    hbiasinc = momentum*hbiasinc + epsilon*dh_total;
    vbiasinc = momentum*vbiasinc + epsilon*dv_total;
    if params.opt_nodeup,
        nodeWinc = momentum*nodeWinc + epsilon*(dnW_total - params.l2reg_node*w_slr.nodeWeights);
    else
        nodeWinc = zeros(size(nodeWinc));
    end
    
    % update parameters
    w_rbm.vishid(:,1:params.numLabel-1,:) = w_rbm.vishid(:,1:params.numLabel-1,:) + vishidinc(:,1:params.numLabel-1,:);
    w_rbm.visbiases(:,1:params.numLabel-1) = w_rbm.visbiases(:,1:params.numLabel-1) + vbiasinc(:,1:params.numLabel-1);
    w_rbm.hidbiases = w_rbm.hidbiases + hbiasinc;
    w_slr.nodeWeights = w_slr.nodeWeights + nodeWinc;
    
    fprintf('epoch %.3d : recon_err = %g\n',t,recon_err_all/params.batchSize);
    save_params(params,w_rbm,w_slr,t,fname_date);
end
fname = sprintf('%s_done',params.fname);
[w_rbm, w_slr] = save_params(params,w_rbm,w_slr,t,fname);

return;

function [w_rbm, w_slr] = save_params(params,w_rbm,w_slr,t,fname_save)
if ~exist('fname_save','var'),
    fname_save = params.fname;
end

w_rbm_old = w_rbm;
clear w_rbm;

w_rbm.vishid = double(w_rbm_old.vishid);
w_rbm.hidbiases = double(w_rbm_old.hidbiases);
w_rbm.visbiases = double(w_rbm_old.visbiases);

w_rbm.vishid(:,params.numLabel,:) = 0;
w_rbm.visbiases(:,params.numLabel) = 0;

w_scrf_old = w_slr;
clear w_slr;

w_slr.nodeWeights = double(w_scrf_old.nodeWeights);
params.maxepoch = t;

save(sprintf('%s/%s.mat',params.savepath,fname_save),'w_rbm','w_slr','params','t');

return;
