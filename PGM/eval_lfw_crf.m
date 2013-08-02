function [acc, evaltime] = eval_lfw_crf(w_crf, datanames, datanums, sds, esds, verbose)
startup_directory;
olddim = 250;
nlabel = size(w_crf.nodeWeights,2);
addpath('model/crf');

tot_err = 0;
tot_sp = 0;
evaltime = 0;
for i = 1:length(datanums),
    % load full data
    gt_casename = sprintf('%s/%s/%s_%04d.dat', gt_dir, datanames{i}, datanames{i}, datanums(i));
    gt_case = load(gt_casename);
    gt_case = gt_case + 1;
    gt_splabels = gt_case(2:end);   % the first value is the number of nodes
    
    % read superpixel features
    [numNodes, H, E, S] = getFeatures(datanames{i}, datanums(i), features_dir);
    X = struct('numNodes', numNodes, 'adjmat', {E}, 'nodeFeatures', {H}, 'edgeFeatures', {S});
    [~, num_sp] = size(X.nodeFeatures);
    
    % scale features
    if w_crf.params.rmposfeat,
        X.nodeFeatures(65:128,:) = [];
    end
    X.nodeFeatures = bsxfun(@rdivide,X.nodeFeatures,sds);
    X.nodeFeatures(end+1,:) = 1;
    
    [xe, ye] = find(X.adjmat > 0);
    for j=1:length(xe)
        X.edgeFeatures{xe(j),ye(j)} = X.edgeFeatures{xe(j),ye(j)} ./ esds;
        X.edgeFeatures{xe(j),ye(j)}(end+1) = 1;
    end
    
    tS = tic;
    labelprob = inference_crf(w_crf, X, nlabel, 0);
    tE = toc(tS);
    evaltime = evaltime + tE;
    
    [~, pred] = max(labelprob ,[], 1);
    err = sum(pred(:) ~= gt_splabels(:));
    tot_err = tot_err + err;
    tot_sp = tot_sp + num_sp;
    if verbose,
        fprintf('[%d/%d] err: %d/%d, acc = %g\n', i,length(datanames),err,num_sp,100*(1-tot_err/tot_sp));
    else
        if ~mod(i,10),
            fprintf('.');
        end
        if ~mod(i,100),
            fprintf('[%d/%d] ',i,length(datanames));
            fprintf('acc = %g\n',100*(1-tot_err/tot_sp));
        end
    end
    clear gt_splabels X sp;
end
evaltime = evaltime/length(datanums);
acc = 100*(1-tot_err/tot_sp);

fprintf('acc = %g, inference time = %g (ex/sec)\n', acc, evaltime);

return