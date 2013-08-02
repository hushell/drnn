% Train GLOC (GLObal and LOCal) model on LFW labeling dataset
% using the pretrained weights from spatial crf and crbm.
%
% input
%   tr_feat         : cell, field - nodeFeatures, edgeFeatures, adjmat, numNodes
%   tr_proj_crf     : cell, {p_{sn}}
%   tr_proj_rbm     : cell, {p_{rs}}
%   tr_label        : cell, numNodes x 1
%   params          : hyperparameters
%   w_scrf          : field - nodeWeights, edgeWeights (pretrained from scrf)
%
% output
%   w_gloc          : field - vishid, hidbiases, visbiases, nodeWeights, edgeWeights
%   w_rbm           : field - vishid, hidbiases, visbiases
%   w_scrf          : field - nodeWeights, edgeWeights
%   params          : hyperparameters
%
%
% reference:
% Augmenting CRFs with Boltzmann Machine Shape Priors for Image Labeling, CVPR, 2013.
%

function [w_gloc, w_rbm, w_scrf, params] = gloc_train_percLoss(tr_feat, tr_proj_crf, tr_proj_rbm, tr_label, params, w_scrf, w_rbm)

warning off all;
close all;
rng('shuffle');

fname_date = sprintf('%s_date%s',params.fname,datestr(now, 30));

initialmomentum  = 0.5;
finalmomentum    = 0.9;

% initialize weight
if ~exist('w_rbm','var') || isempty(w_rbm),
    w_rbm.vishid = 0.1*randn(params.numNodes_rbm, params.numLabel-1, params.numHid);
    w_rbm.visbiases = zeros(params.numNodes_rbm,params.numLabel-1);
    w_rbm.hidbiases = zeros(params.numHid,1);
end
if size(w_rbm.vishid,2) < params.numLabel,
    w_rbm.vishid(:,size(w_rbm.vishid,2)+1:params.numLabel,:) = 0;
end
if size(w_rbm.visbiases,2) < params.numLabel,
    w_rbm.visbiases(:,size(w_rbm.visbiases,2)+1:params.numLabel) = 0;
end

if ~exist('w_scrf','var') || isempty(w_scrf),
    w_scrf.nodeWeights = 0.1*randn(params.numDim,params.numLabel,params.numNodes_crf);
    w_scrf.edgeWeights = 0.1*randn(4,params.numLabel,params.numLabel);
end

vishidinc = zeros(size(w_rbm.vishid));
vbiasinc = zeros(size(w_rbm.visbiases));
hbiasinc = zeros(size(w_rbm.hidbiases));
nodeWinc = zeros(size(w_scrf.nodeWeights));
edgeWinc = zeros(size(w_scrf.edgeWeights));

disp(params);
if params.nMF == 0,
    nMF = 200;
end

%%% --- learning --- %%%
mu_node = cell(params.batchSize,1);
label_mult = cell(params.batchSize,1);
neglabel = cell(params.batchSize,1);
label_proj = single(zeros(params.numNodes_rbm,params.numLabel,params.batchSize));
edgeFeat = cell(params.batchSize,1);

for j = 1:params.batchSize,
    label = tr_label{j};
    label_mult{j} = multi_output(label,params.numLabel)'; % numDim x numCh
end

for t = 1:params.maxepoch,
    % momentum update
    if t < params.momch, momentum = initialmomentum;
    else momentum = finalmomentum; end
    
    % # CD steps, learning rate
    if params.anneal,
        KCD = min(max(ceil(t/10),5),30);
        epsilon = params.epsilon/(1+params.epsdecay*KCD);
    else
        KCD = params.KCD;
        epsilon = params.epsilon/(1+params.epsdecay*t);
    end
    
    % reshape variables to speed up
    w_rbm.vishidrs = reshape(w_rbm.vishid,size(w_rbm.vishid,1)*size(w_rbm.vishid,2),size(w_rbm.vishid,3));
    w_rbm.visbiasrs  = reshape(w_rbm.visbiases,size(w_rbm.vishid,1)*size(w_rbm.vishid,2),1);
    w_scrf.nodeWeightsrs = reshape(w_scrf.nodeWeights,size(w_scrf.nodeWeights,1)*size(w_scrf.nodeWeights,2),size(w_scrf.nodeWeights,3));
    
    tS = tic;
    
    %%% ----- positive phase ----- %%%
    %%% hidden unit inference
    for j = 1:params.batchSize,
        proj_rbm = tr_proj_rbm{j};
        label_proj(:,:,j) = proj_rbm*label_mult{j};
    end
    ey = bsxfun(@plus,w_rbm.hidbiases,w_rbm.vishidrs'*reshape(label_proj,params.numNodes_rbm*params.numLabel,params.batchSize));
    poshidprob = sigmoid(ey);
    
    % positive gradient
    dvh_pos = reshape(reshape(label_proj,params.numNodes_rbm*params.numLabel,params.batchSize)*poshidprob',size(w_rbm.vishid))/params.batchSize;
    dh_pos = mean(poshidprob,2);
    dv_pos = mean(label_proj,3);
    
    %%% visible unit inference (not necessary for percLoss)
    % energy from rbm
    
    mu_rbm = reshape(bsxfun(@plus,w_rbm.vishidrs*poshidprob,w_rbm.visbiasrs),params.numNodes_rbm,params.numLabel,params.batchSize);
    recon_err_all = 0;
    for j = 1:params.batchSize,
        feat = tr_feat{j};
        proj_rbm = tr_proj_rbm{j};
        proj_crf = tr_proj_crf{j};
        
        if ~isfield(feat,'edgeFeaturesrs'),
            [xi, xj] = find(feat.adjmat > 0);
            edgeFeaturesrs = zeros(size(w_scrf.edgeWeights,1),1,1,length(xi));
            for fi = 1:length(xi),
                edgeFeaturesrs(:,:,:,fi) = feat.edgeFeatures{xi(fi), xj(fi)};
            end
            feat.edgeFeaturesrs = single(edgeFeaturesrs);
            tr_feat{j} = feat;
        end
        
        if isfield(feat,'nodeFeatures'),
            feat.nodeFeaturesrs = permute(feat.nodeFeatures,[1 3 2]);
            feat = rmfield(feat,'nodeFeatures');
            tr_feat{j} = feat;
        end
        
        if ~isfield(feat,'Aimat'),
            feat = unfold_edgeFeat(feat);
            tr_feat{j} = feat;
        end
        
        % node potential
        mu_node{j} = single(squeeze(sum(bsxfun(@times,feat.nodeFeaturesrs,reshape(w_scrf.nodeWeightsrs*proj_crf,params.numDim,params.numLabel,feat.numNodes)),1)));
        
        % edge feature
        AiWe = sum(bsxfun(@times,feat.Aimat,w_scrf.edgeWeights),1);
        AjWe = sum(bsxfun(@times,feat.Ajmat,w_scrf.edgeWeights),1);
        cedgeFeat = zeros(params.numLabel,params.numLabel,feat.numNodes,feat.numNodes);
        for ei = 1:feat.numNodes,
            cedgeFeat(:,:,ei,feat.Aji{ei}) = reshape(AiWe(:,:,:,feat.Aiidx{ei}),params.numLabel,params.numLabel,1,feat.Ailen{ei});
            cedgeFeat(:,:,ei,feat.Aij{ei}) = reshape(AjWe(:,:,:,feat.Ajidx{ei}),params.numLabel,params.numLabel,1,feat.Ajlen{ei});
        end
        
        cedgeFeat = permute(cedgeFeat,[1 3 2 4]);
        edgeFeat{j} = single(reshape(cedgeFeat,size(cedgeFeat,1)*size(cedgeFeat,2),size(cedgeFeat,3)*size(cedgeFeat,4)));
        
        
        % aggregate energy from node and rbm
        mu_ey = mu_node{j} + mu_rbm(:,:,j)'*proj_rbm;
        
        % initialize with lr + rbm
        mu_max = max(mu_ey,[],1);
        mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu_ey,mu_max)),sum(exp(bsxfun(@minus,mu_ey,mu_max)),1));
        
        % mean-field iteration (full block)
        mu_old = mu;
        for nmf = 1:nMF,
            % mu: nLabels x numFeat
            mu = mu_ey + reshape(edgeFeat{j}*mu(:),params.numLabel,size(edgeFeat{j},1)/params.numLabel);
            
            mu_max = max(mu,[],1);
            mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu,mu_max)),sum(exp(bsxfun(@minus,mu,mu_max)),1));
            err = norm(mu(:) - mu_old(:));
            if err < 1e-4,
                break;
            else
                mu_old = mu;
            end
        end
        neglabel{j} = mu';
        recon_err = norm(label_mult{j} - neglabel{j});
        recon_err_all = recon_err_all + recon_err;
    end
    
    
    %%% ----- negative phase ----- %%%
    seg_err_list = zeros(params.batchSize,1);
    num_sp_list = zeros(params.batchSize,1);
    fprintf('start negative phase ... ');
    for j = 1:params.batchSize,
        feat = tr_feat{j};
        proj_crf = tr_proj_crf{j};
        proj_rbm = tr_proj_rbm{j};
        mu_node_cur = mu_node{j};
        
        % initialize with lr
        mu_max = max(mu_node_cur,[],1);
        mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu_node_cur,mu_max)),sum(exp(bsxfun(@minus,mu_node_cur,mu_max)),1));
        
        %%% hidden unit inference
        label_proj_cur = proj_rbm*mu';
        ey = bsxfun(@plus,w_rbm.hidbiases,w_rbm.vishidrs'*reshape(label_proj_cur,params.numNodes_rbm*params.numLabel,1));
        neghidprob = sigmoid(ey);
        neghidstate = rand(size(neghidprob)) < neghidprob;
        
        %%% CD step (1 step of mean-field)
        fey = inf;
        prodFWP = [];
        prodFW = [];
        prodPC = [];
        prodPW = [];
        
        for kcd = 1:KCD,
            %%% visible unit inference
            % energy from w_rbm
            mu_rbm = reshape(bsxfun(@plus,w_rbm.vishidrs*neghidstate,w_rbm.visbiasrs),params.numNodes_rbm,params.numLabel);
            
            % aggregate energy from node and rbm
            mu_ey = mu_node_cur + mu_rbm'*proj_rbm;
            
            % mu: nLabels x numFeat
            mu = mu_ey + reshape(edgeFeat{j}*mu(:),params.numLabel,size(edgeFeat{j},1)/params.numLabel);
            
            mu_max = max(mu,[],1);
            mu = bsxfun(@rdivide,exp(bsxfun(@minus,mu,mu_max)),sum(exp(bsxfun(@minus,mu,mu_max)),1));
            
            [fey_cur, prodFWP, prodFW, prodPC, prodPW] = fey_gloc(mu, feat, w_scrf, w_rbm, params, proj_crf, proj_rbm, prodFWP, prodFW, prodPC, prodPW);
            if fey_cur < fey,
                mu_lowest_ey = mu;
                fey = fey_cur;
            end
            
            %%% hidden unit inference
            label_proj_cur = proj_rbm*mu';
            ey = bsxfun(@plus,w_rbm.hidbiases,w_rbm.vishidrs'*reshape(label_proj_cur,params.numNodes_rbm*params.numLabel,1));
            neghidprob = sigmoid(ey);
            neghidstate = rand(size(neghidprob)) < neghidprob;
        end
        mu = mu_lowest_ey;
        neglabel{j} = mu';
        
        [~,pred] = max(mu,[],1);
        [~,gt] = max(label_mult{j},[],2);
        seg_err_list(j) = sum(pred(:) ~= gt(:));
        num_sp_list(j) = feat.numNodes;
        label_proj(:,:,j) = proj_rbm*mu';
    end
    fprintf('done!\n');
    seg_err = sum(seg_err_list);
    num_sp = sum(num_sp_list);
    
    %%% hidden unit inference
    ey = bsxfun(@plus,w_rbm.hidbiases,w_rbm.vishidrs'*reshape(label_proj,params.numNodes_rbm*params.numLabel,params.batchSize));
    neghidprob = sigmoid(ey);
    
    % negative gradient
    dvh_neg = reshape(reshape(label_proj,params.numNodes_rbm*params.numLabel,params.batchSize)*neghidprob',size(w_rbm.vishid))/params.batchSize;
    dh_neg = mean(neghidprob,2);
    dv_neg = mean(label_proj,3);
    
    % gradients
    dvh_total = dvh_pos - dvh_neg;
    dh_total = dh_pos - dh_neg;
    dv_total = dv_pos - dv_neg;
    
    dnW_total = zeros(size(w_scrf.nodeWeights));
    k = 0;
    for j = 1:params.batchSize,
        feat = tr_feat{j};
        proj_crf = tr_proj_crf{j};
        dnW_total = dnW_total + reshape(reshape(bsxfun(@times,feat.nodeFeaturesrs,permute(label_mult{j}-neglabel{j},[3 2 1])),params.numDim*params.numLabel,feat.numNodes)*proj_crf',params.numDim,params.numLabel,params.numNodes_crf);
        k = k + feat.numNodes;
    end
    dnW_total = dnW_total/k;
    
    deW_total = zeros(size(w_scrf.edgeWeights));
    k = 0;
    for j = 1:params.batchSize,
        feat = tr_feat{j};
        [xi, xj] = find(feat.adjmat > 0);
        deW_total = deW_total + 0.5*sum(bsxfun(@times,bsxfun(@times,feat.edgeFeaturesrs,permute(label_mult{j}(xi,:)',[3 1 4 2])),permute(label_mult{j}(xj,:)',[3 4 1 2])),4);
        deW_total = deW_total + 0.5*sum(bsxfun(@times,bsxfun(@times,feat.edgeFeaturesrs,permute(label_mult{j}(xj,:)',[3 1 4 2])),permute(label_mult{j}(xi,:)',[3 4 1 2])),4);
        deW_total = deW_total - 0.5*sum(bsxfun(@times,bsxfun(@times,feat.edgeFeaturesrs,permute(neglabel{j}(xi,:)',[3 1 4 2])),permute(neglabel{j}(xj,:)',[3 4 1 2])),4);
        deW_total = deW_total - 0.5*sum(bsxfun(@times,bsxfun(@times,feat.edgeFeaturesrs,permute(neglabel{j}(xj,:)',[3 1 4 2])),permute(neglabel{j}(xi,:)',[3 4 1 2])),4);
        k = k + feat.numNodes;
    end
    deW_total = deW_total/k;
    
    % gradient update
    vishidinc = momentum*vishidinc + epsilon*(dvh_total - params.l2reg*w_rbm.vishid);
    hbiasinc = momentum*hbiasinc + epsilon*dh_total;
    vbiasinc = momentum*vbiasinc + epsilon*dv_total;
    nodeWinc = momentum*nodeWinc + (k/params.batchSize)*epsilon*(dnW_total - params.l2reg_node*w_scrf.nodeWeights);
    edgeWinc = momentum*edgeWinc + (k/params.batchSize)*epsilon*(deW_total - params.l2reg_edge*w_scrf.edgeWeights);
    
    % update parameters
    w_rbm.vishid(:,1:params.numLabel-1,:) = w_rbm.vishid(:,1:params.numLabel-1,:) + vishidinc(:,1:params.numLabel-1,:);
    w_rbm.visbiases(:,1:params.numLabel-1) = w_rbm.visbiases(:,1:params.numLabel-1) + vbiasinc(:,1:params.numLabel-1);
    w_rbm.hidbiases = w_rbm.hidbiases + hbiasinc;
    w_scrf.nodeWeights = w_scrf.nodeWeights + nodeWinc;
    w_scrf.edgeWeights = w_scrf.edgeWeights + edgeWinc;
    
    tE = toc(tS);
    
    fprintf('epoch %.3d : recon_err = %g, pred_acc = %g, time = %g\n',t,recon_err_all/params.batchSize,1-seg_err/num_sp,tE);
    fprintf('filepath: %s\n',params.savepath);
    fprintf('filename: %s\n\n',params.fname);
    
    save_params(params,w_rbm,w_scrf,t);
    if ismember(t,params.save_epochs),
        fname = sprintf('%s_%04d',params.fname,t);
        save_params(params,w_rbm,w_scrf,t,fname);
    end
    save_params(params,w_rbm,w_scrf,t,fname_date);
end
fname = sprintf('%s_%04d',params.fname,params.maxepoch);
[w_rbm, w_scrf, w_gloc] = save_params(params,w_rbm,w_scrf,t,fname);

return

function [w_rbm, w_scrf, w_gloc] = save_params(params,w_rbm,w_scrf,t,fname_save)
if ~exist('fname_save','var'),
    fname_save = params.fname;
end

params.fname = fname_save;
w_rbm_old = w_rbm;
clear w_rbm;

w_rbm.vishid = double(w_rbm_old.vishid);
w_rbm.hidbiases = double(w_rbm_old.hidbiases);
w_rbm.visbiases = double(w_rbm_old.visbiases);

w_rbm.vishid(:,params.numLabel,:) = 0;
w_rbm.visbiases(:,params.numLabel) = 0;

w_scrf_old = w_scrf;
clear w_scrf;

w_scrf.nodeWeights = double(w_scrf_old.nodeWeights);
w_scrf.edgeWeights = double(w_scrf_old.edgeWeights);

w_gloc.vishid = w_rbm.vishid;
w_gloc.hidbiases = w_rbm.hidbiases;
w_gloc.visbiases = w_rbm.visbiases;
w_gloc.nodeWeights = w_scrf.nodeWeights;
w_gloc.edgeWeights = w_scrf.edgeWeights;

save(sprintf('%s/%s.mat',params.savepath,fname_save),'w_gloc','w_rbm','w_scrf','params','t');

return;
