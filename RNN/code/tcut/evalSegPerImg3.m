function [PRs,spErrs,nCuts,PRs3,GCEs,VIs] = evalSegPerImg3(...
  name, tree_cut_func, imgData, imgTreeTop, theta_plus, n_labs, lamb_samples, vis, vtree)
% evaluation segmentation per image, try different p_connect

if nargin < 8
    vis = 0;
    vtree = 0;
end

if nargin < 7
    vtree = 0;
    vis = 1;
    lamb_samples = 0;
end

if nargin < 6
    vtree = 0;
    vis = 0;
    n_labs = 8;
    lamb_samples = 0;
end

if ~isempty(name)
  addpath ../../../vis/export_fig/
end

colorCat = zeros(size(imgData.segs2));
connectCompGT = zeros(size(imgData.segs2));
forestLabels = zeros(size(imgData.segs2));
spLabelGT = zeros(size(imgData.segs2));

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
    
    vl_tightsubplot(1+3,r,'box','outer'); 
    imagesc(imgData.img); 
    %imagesc(imgData.labels);
    title('image'); 
    axis off
    r = r + 1;
    
    vl_tightsubplot(1+3,r,'box','outer'); 
    imagesc(connectCompGT); 
    %imagesc(imgData.labels);
    title('connected component gt'); 
    axis off
    r = r + 1;
    
    vl_tightsubplot(1+3,r,'box','outer'); 
    imagesc(spLabelGT);
    title('superpixel gt'); 
    axis off
    r = r + 1;
end

% for each lambda (p_connect), run tree_cut_postpone()
if 1
    lambda = lamb_samples;
    [Q,cuts,labs,forest] = tree_cut_func(imgData, imgTreeTop, theta_plus, n_labs, lambda);
    labels = labs(1:length(imgData.segLabels));
    nCuts(j) = sum(cuts);
    
    for i = 1:length(imgData.segLabels)
        colorCat(imgData.segs2 == i) = labels(i);
        forestLabels(imgData.segs2 == i) = forest(i);
    end
    connectCompGT = connectCompGT + 1;
    [PRs(j), GCEs(j), VIs(j)] = compare_segmentations(forestLabels, connectCompGT);
    %[PRs2(j), ~] = match_segmentations2(forestLabels, {connectCompGT});
    [PRs3(j)] = adjrand(forestLabels(:), connectCompGT(:));
    %PASCALs(j) = match_segmentations({colorCat}, {spLabelGT});
    spErrs(j) = sum(labels == imgData.segLabels) / length(labels);
    
    combo(j) = (PRs(j).*spErrs(j).*PRs3(j)) ./ (nCuts(j).*(VIs(j)+1e-5));
 
    if vis == 1
        vl_tightsubplot(1+3,r,'box','outer'); 
        %imagesc(colorCat); title(sprintf('p = %f', lambda));
        %imagesc(forestLabels); title(sprintf('p = %.5f, PR = %.5f', lambda, PRs(j)));
        imagesc(forestLabels); title(sprintf('p = %.5f, all = %.5f', lambda, combo(j)));
        axis off
        r = r + 1;
    end
end
fprintf('------ GT: number of Connected Component = %d\n', numel(unique(connectCompGT)));

if vtree
  ht = figure(103); 
  imgTreeTop.plotForest(forest); 
  set(ht, 'Position', get(0,'Screensize'));
  title([name(1:6) ': num(forest) = ' num2str(length(unique(forest)))]);
  export_fig([name '_forest'], '-eps')
end

if vis == 1
    if ~isempty(name)
      h100 = figure(100);
      set(h100, 'Position', get(0,'Screensize'));
      export_fig([name '_vis'], '-eps')
    end
end