function [PRs,spErrs,nCuts] = evalSegPerImg(name, tree_cut_func, imgData, imgTreeTop, theta_plus, n_labs, lamb_samples, vis, vtree)
% evaluation segmentation per image, try different p_connect

if nargin < 8
    vis = 0;
    vtree = 0;
end

if nargin < 7
    vtree = 0;
    vis = 1;
    lamb_samples = [0:0.05:1];
end

if nargin < 6
    vtree = 0;
    vis = 0;
    n_labs = 8;
    lamb_samples = [0:0.05:1];
end

if ~isempty(name)
  addpath ../../../vis/export_fig/
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

% get SP GT
for i = 1:length(imgData.segLabels)
    spLabelGT(imgData.segs2 == i) = imgData.segLabels(i);
end

% get connected component label
clabs = 1:300;
cci = 1;
for i = 1:n_labs
    %temp = bwlabel(imgData.labels == i); 
    temp = bwlabel(spLabelGT == i);
    vcc = unique(temp(temp > 0));
    %temp(temp > 0) = temp(temp > 0) + i*n_labs;
    for k = 1:numel(vcc)
      temp(temp == vcc(k)) = clabs(cci);
      cci = cci + 1;
    end
    connectCompGT = connectCompGT + temp;
end

j = 1;
r = 1;

%vis = 0;
if vis == 1
    figure(100);
    clf;
    
    vl_tightsubplot(numel(lamb_samples)+3,r,'box','outer'); 
    imagesc(imgData.img); 
    %imagesc(imgData.labels);
    title('image'); 
    axis off
    r = r + 1;
    
    vl_tightsubplot(numel(lamb_samples)+3,r,'box','outer'); 
    imagesc(connectCompGT); 
    %imagesc(imgData.labels);
    title('connected component gt'); 
    axis off
    r = r + 1;
    
    vl_tightsubplot(numel(lamb_samples)+3,r,'box','outer'); 
    imagesc(spLabelGT);
    title('superpixel gt'); 
    axis off
    r = r + 1;
end

% for each lambda (p_connect), run tree_cut_postpone()
for lambda = lamb_samples
    [Q,cuts,labs,forest] = tree_cut_func(imgData, imgTreeTop, theta_plus, n_labs, lambda);
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
 
    if vis == 1
        vl_tightsubplot(numel(lamb_samples)+3,r,'box','outer'); 
        %imagesc(colorCat); title(sprintf('p = %f', lambda));
        imagesc(forestLabels); title(sprintf('p = %.5f, PR = %.5f', lambda, PRs(j)));
        axis off
        r = r + 1;
    end
    j = j + 1;
end
fprintf('------ GT: number of Connected Component = %d\n', numel(unique(connectCompGT)));

if vtree
  [peak,loc] = fullmax(PRs);
  [mnc,mmi] = min(nCuts(loc));
  [Q,cuts,labs,forest] = tree_cut_func(imgData, imgTreeTop, theta_plus, n_labs, lamb_samples(loc(mmi)));
  ht = figure(103); 
  imgTreeTop.plotForest(forest); 
  set(ht, 'Position', get(0,'Screensize'));
  title([name ': num(forest) = ' num2str(length(unique(forest)))]);
  export_fig([name '_forest'], '-eps')
end

if vis == 1
    if ~isempty(name)
      h100 = figure(100);
      set(h100, 'Position', get(0,'Screensize'));
      export_fig([name '_vis'], '-eps')
    end
    figure(101);
    subplot(2,3,1); plot(lamb_samples, PRs, '-go', 'linewidth', 2); legend('PR');
    subplot(2,3,2); plot(lamb_samples, GCEs, '-bo', 'linewidth', 2); legend('GCE');
    subplot(2,3,3); plot(lamb_samples, VIs, '-ro', 'linewidth', 2); legend('VI');
    subplot(2,3,4); plot(lamb_samples, spErrs, '-ro', 'linewidth', 2); legend('spAcc');
    subplot(2,3,5); plot(lamb_samples, nCuts, '-bo', 'linewidth', 2); legend('nCuts');
    if ~isempty(name)
      h101 = figure(101);
      set(h101, 'Position', get(0,'Screensize'));
      export_fig([name '_cur'], '-eps')
    end
end