function [PRs,spErrs,nCuts] = evalSegPerImg(imgData, imgTreeTop, theta_plus, n_labs, lamb_samples, vis)
% evaluation segmentation per image, try different p_connect

if nargin < 6
    vis = 0;
end

if nargin < 5
    vis = 1;
    lamb_samples = [0:0.05:1];
end

if nargin < 4
    vis = 0;
    n_labs = 8;
    lamb_samples = [0:0.05:1];
end

colorCat = zeros(size(imgData.segs2));
connectCompGT = zeros(size(imgData.segs2));
forestLabels = zeros(size(imgData.segs2));
spLabelGT = zeros(size(imgData.segs2));

PRs = zeros(size(lamb_samples));
GCEs = zeros(size(lamb_samples));
VIs = zeros(size(lamb_samples));
nCuts = zeros(size(lamb_samples));
PRs2 = zeros(size(lamb_samples));
PASCALs = zeros(size(lamb_samples));
spErrs = zeros(size(lamb_samples));

% get connected component label
for i = 1:n_labs
    temp = bwlabel(imgData.labels == i); 
    temp(temp > 0) = temp(temp > 0) + i;
    connectCompGT = connectCompGT + temp;
end

j = 1;
r = 1;

%vis = 0;
if vis == 1
    figure(100);
    clf;
    vl_tightsubplot(numel(lamb_samples)+2,r,'box','outer'); 
    %imagesc(connectCompGT); 
    imagesc(imgData.labels);
    title('gt'); 
    axis off
    r = r + 1;
    
    for i = 1:length(imgData.segLabels)
        spLabelGT(imgData.segs2 == i) = imgData.segLabels(i);
    end
    
    vl_tightsubplot(numel(lamb_samples)+2,r,'box','outer'); 
    imagesc(spLabelGT);
    title('sp gt'); 
    axis off
    r = r + 1;
end

% for each lambda (p_connect), run tree_cut_postpone()
for lambda = lamb_samples
    %[Q,cuts,labels,forest] = tree_cut_postpone(imgData, imgTreeTop, theta_plus, n_labs, lambda);
    [Q,cuts,labs,forest] = tree_cut_new(imgData, imgTreeTop, theta_plus, n_labs, lambda);
    labels = labs(1:length(imgData.segLabels));
    nCuts(j) = sum(cuts);
    
    for i = 1:length(imgData.segLabels)
        colorCat(imgData.segs2 == i) = labels(i);
        forestLabels(imgData.segs2 == i) = forest(i);
    end
    [PRs(j), GCEs(j), VIs(j)] = compare_segmentations(forestLabels, connectCompGT);
    %[PRs2(j), ~] = match_segmentations2(colorCat, {connectCompGT});
    %PASCALs(j) = match_segmentations(colorCat, {connectCompGT});
    spErrs(j) = sum(labels == imgData.segLabels) / length(labels);
    j = j + 1;
    
    fprintf('------ GT number of Connected Component = %d\n', numel(unique(connectCompGT)));
 
    if vis == 1
        vl_tightsubplot(numel(lamb_samples)+2,r,'box','outer'); 
        imagesc(colorCat); title(sprintf('p = %f', lambda));
        axis off
        r = r + 1;
    end
end

if vis == 1
    figure(101);
    subplot(2,3,1); plot(lamb_samples, PRs, '-go', 'linewidth', 2); legend('PR');
    subplot(2,3,2); plot(lamb_samples, GCEs, '-bo', 'linewidth', 2); legend('GCE');
    subplot(2,3,3); plot(lamb_samples, VIs, '-ro', 'linewidth', 2); legend('VI');
    subplot(2,3,4); plot(lamb_samples, spErrs, '-ro', 'linewidth', 2); legend('spErr');
    subplot(2,3,5); plot(lamb_samples, nCuts, '-bo', 'linewidth', 2); legend('nCuts');
end