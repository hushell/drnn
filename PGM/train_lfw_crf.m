%%% train conditional random fields on LFW labeling dataset
fprintf('\nstart CRF training\n\n');
fprintf('processing the features!!\n');


Y = cell(length(trainnames),1);
for i = 1:length(trainnames),
    gtfn = sprintf('%s/%s/%s_%04d.dat', gt_dir, trainnames{i}, trainnames{i}, trainnums(i));
    fidgt = fopen(gtfn);
    numNodes = fscanf(fidgt, '%d', 1);
    Y{i} = fscanf(fidgt, '%d', [1 numNodes]) + 1;
    fclose(fidgt);
end
clear X;

for i = 1:length(trainnames),
    if ~mod(i,10), fprintf('.'); end
    if ~mod(i,500), fprintf('%d\n',i); end
    % superpixel features
    % numNodes  : number of superpixels
    % H         : node features
    % E         : adjacent matrix
    % S         : edge features
    [numNodes, H, E, S] = getFeatures(trainnames{i}, trainnums(i), features_dir);
    X(i) = struct('numNodes', numNodes, 'adjmat', {E}, 'nodeFeatures', {H}, 'edgeFeatures', {S});
    clear H E S numNodes;
end

for i = 1:length(X),
    X(i).nodeFeatures = bsxfun(@rdivide,X(i).nodeFeatures,sds);
    % add bias term
    X(i).nodeFeatures(end+1,:) = 1;
end

for i = 1:length(X),
    [xe, ye] = find(X(i).adjmat > 0);
    for j = 1:length(xe),
        X(i).edgeFeatures{xe(j),ye(j)} = X(i).edgeFeatures{xe(j),ye(j)} ./ esds;
        % add bias term
        X(i).edgeFeatures{xe(j),ye(j)}(end+1) = 1;
    end
end


%%% --- Train CRF with MeanField --- %%%
w_crf.nodeWeights = w_lr.nodeWeights; % initialize from lr
params = w_crf.params;

w_crf = crf_train(w_crf, Y, X, nlabel, l2reg_node, l2reg_edge);
w_crf.params = params;

clear X Y;
