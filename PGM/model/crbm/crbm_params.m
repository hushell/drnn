function params = crbm_params(dataSet,fname_save,numDim,numLabel,dim_crf,l2reg_node,...
    dim_rbm,opt_nodeup,numHid,l2reg,epsilon,KCD,anneal,batchSize,maxepoch,savepath)

% set crbm_full parameters
% data specific
if ~exist('numDim','var') || isempty(numDim),
    numDim = 193;
end
if ~exist('numLabel','var') || isempty(numLabel),
    numLabel = 3;
end

% lr
if ~exist('dim_crf','var') || isempty(dim_crf),
    dim_crf = 16;
end
if ~exist('l2reg_node','var') || isempty(l2reg_node),
    l2reg_node = 3e-5;
end

% rbm
if ~exist('dim_rbm','var') || isempty(dim_rbm),
    dim_rbm = 24;
end
if ~exist('opt_nodeup','var') || isempty(opt_nodeup),
    opt_nodeup = 0;
end
if ~exist('numHid','var') || isempty(numHid),
    numHid = 400;
end
if ~exist('l2reg','var') || isempty(l2reg),
    l2reg = 1e-4;
end
if ~exist('epsilon','var') || isempty(epsilon),
    epsilon = 0.01;
end
if ~exist('KCD','var') || isempty(KCD),
    KCD = 1;
end
if ~exist('maxepoch','var') || isempty(maxepoch),
    maxepoch = 100;
end
if ~exist('batchSize','var') || isempty(batchSize),
    batchSize = 0;
end
if ~exist('savepath','var') || isempty(savepath),
    savepath = 'results';
end
if ~exist('anneal','var') || isempty(anneal),
    anneal = 0;
end

params.dataSet = dataSet;
params.numDim = numDim;
params.numLabel = numLabel;

params.numNodes_crf = dim_crf^2;
params.l2reg_node = l2reg_node;

params.numNodes_rbm = dim_rbm^2;
params.opt_nodeup = opt_nodeup;
params.numHid = numHid;
params.l2reg = l2reg;
params.epsilon = epsilon;
params.epsdecay = 0.01;
params.KCD = KCD;
params.anneal = anneal;
params.maxepoch = maxepoch;
params.batchSize = batchSize;
params.savepath = savepath;
params.momch = 5;

if ~exist(params.savepath,'dir'),
    mkdir(params.savepath);
end

if ~exist('fname_save','var') || isempty(fname_save),
    params.fname = sprintf('crbm_%s_nD%d_nL%d_N%d_l2n%g_rbm_R%d_n%d_nH%d_l2r%g_eps%g_CD%d_ann%d_bS%d_iter%d',dataSet,numDim,nlabel,dim_crf,l2reg_node,...
        dim_rbm,opt_nodeup,numHid,l2reg,epsilon,KCD,anneal,batchSize,maxepoch);
else
    params.fname = fname_save;
end

return;
