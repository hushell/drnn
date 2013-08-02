function [acc, evaltime] = eval_lfw_lr(w_lr, datanames, datanums, sds, verbose)
startup_directory;
olddim = 250;
nlabel = size(w_lr.nodeWeights,2);
addpath('model/lr');

tot_err = 0;
tot_sp = 0;
evaltime = 0;
for i = 1:length(datanames),
    % load full data
    gt_casename = sprintf('%s/%s/%s_%04d.dat', gt_dir, datanames{i}, datanames{i}, datanums(i));
    gt_case = load(gt_casename);
    gt_case = gt_case + 1;
    gt_splabels = gt_case(2:end);   % the first value is the number of nodes
    
    % read superpixel features
    [~, H , ~, ~] = getFeatures(datanames{i}, datanums(i), features_dir);
    [~, num_sp] = size(H);
    if w_lr.params.rmposfeat,
        H(65:128,:) = [];
    end
    whiteH = H ./ repmat(sds, [1, num_sp]); % whiten the features
    feat = whiteH';
    numFeat = size(feat, 1);
    feat = [feat ones(numFeat, 1)]; % add bias term
    feat = feat';
    
    tS = tic;
    labelprob = inference_lr(feat, w_lr.nodeWeights);
    tE = toc(tS);
    evaltime = evaltime + tE;
    
    [~, pred] = max(labelprob ,[], 1);
    err = sum(pred(:) ~= gt_splabels(:));
    tot_err = tot_err + err;
    tot_sp = tot_sp + num_sp;
    if verbose,
        fprintf('valid: [%d/%d] err: %d/%d, acc = %g\n', i,length(datanames),err,num_sp,100*(1-tot_err/tot_sp));
    else
        if ~mod(i,10),
            fprintf('.');
        end
        if ~mod(i,100),
            fprintf('[%d/%d] ',i,length(datanames));
            fprintf('acc = %g\n',100*(1-tot_err/tot_sp));
        end
    end
    clear gt_splabels sp feat;
end
evaltime = evaltime/length(datanames);
acc = 100*(1-tot_err/tot_sp);

fprintf('acc = %g, inference time = %g (ex/sec)\n', acc, evaltime);

return