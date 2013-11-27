function evalSegPerImg(imgData, imgTreeTop, theta_plus, n_labs, lamb_samples, vis)

if nargin < 6
    vis = 0;
end

if nargin < 5
    vis = 0;
    lamb_samples = [0:0.05:1];
end

if nargin < 4
    vis = 0;
    n_labs = 8;
    lamb_samples = [0:0.05:1];
end

colorCat = zeros(size(imgData.segs2));
spColorGT = zeros(size(imgData.segs2));
for i = 1:n_labs
    temp = bwlabel(imgData.labels == i); 
    temp(temp > 0) = temp(temp > 0) + i;
    spColorGT = spColorGT + temp;
end

PRs = zeros(size(lamb_samples));
GCEs = zeros(size(lamb_samples));
VIs = zeros(size(lamb_samples));
nCuts = zeros(size(lamb_samples));
PRs2 = zeros(size(lamb_samples));
PASCALs = zeros(size(lamb_samples));
spErrs = zeros(size(lamb_samples));

j = 1;

%vis = 0;
if vis == 1
    figure;
    vl_tightsubplot(numel(lamb_samples)+1,j,'box','outer'); 
    imagesc(spColorGT); title('gt'); 
    axis off
end
for lambda = lamb_samples
    [Q,cuts,labels,forest] = tree_cut(imgData, imgTreeTop, theta_plus, n_labs, lambda);
    nCuts(j) = sum(cuts);
    
    for i = 1:length(imgData.segLabels)
        colorCat(imgData.segs2 == i) = forest(i);
    end
    [PRs(j), GCEs(j), VIs(j)] = compare_segmentations(colorCat, spColorGT);
    %[PRs2(j), ~] = match_segmentations2(colorCat, {spColorGT});
    %PASCALs(j) = match_segmentations(colorCat, {spColorGT});
    spErrs(j) = sum(labels == imgData.segLabels) / length(labels);
    
    j = j + 1;
 
    if vis == 1
        vl_tightsubplot(numel(lamb_samples)+1,j,'box','outer'); 
        imagesc(colorCat); title(sprintf('p = %f', lambda));
        axis off
    end
end

if vis == 1
    figure;
    subplot(1,3,1); plot(lamb_samples, PRs, '-go', 'linewidth', 2); legend('PR');
    %subplot(1,2,1); plot(lamb_samples, GCEs, '-bo', 'linewidth', 2); legend('GCE');
    %subplot(1,2,1); plot(lamb_samples, VIs, '-ro', 'linewidth', 2); legend('VI');
    subplot(1,3,2); plot(lamb_samples, spErrs, '-ro', 'linewidth', 2); legend('spErr');
    subplot(1,3,3); plot(lamb_samples, nCuts, '-bo', 'linewidth', 2); legend('nCuts');
end