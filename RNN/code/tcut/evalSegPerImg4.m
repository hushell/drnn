function forest_samples = evalSegPerImg4(...
  name, tree_cut_func, imgData, imgTreeTop, theta_plus, n_labs, lamb_samples, vis, vtree)

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

%[Q,cuts,labs,forest] = tree_cut_func(imgData, imgTreeTop, theta_plus, n_labs, lamb_samples);

spLabelGT = zeros(size(imgData.segs2));
connectCompGT = zeros(size(imgData.segs2));

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

nfores = 10;
forest_samples = cell(nfores,1);

if vis
    totp = nfores+2;
    r = 1;
    figure(100);
    clf;
    
    vl_tightsubplot(totp,r,'box','outer'); 
    imagesc(imgData.img); 
    %imagesc(imgData.labels);
    title('image'); 
    axis off
    r = r + 1;
    
    vl_tightsubplot(totp,r,'box','outer'); 
    imagesc(connectCompGT); 
    %imagesc(imgData.labels);
    title('connected component gt'); 
    axis off
    r = r + 1;
    
    for k = 1:nfores
      forest = tree_cut_sample(imgTreeTop, lamb_samples);
      forest_samples{k} = forest;
      forestLabels = zeros(size(imgData.segs2));
      for i = 1:length(imgData.segLabels)
        forestLabels(imgData.segs2 == i) = forest(i);
      end
      vl_tightsubplot(totp,r,'box','outer'); 
      imagesc(forestLabels);
      title(['forest sample ' num2str(k)]); 
      axis off
      r = r + 1;
    end
end

if vtree
%   ht = figure(103); 
%   imgTreeTop.plotForest(forest); 
%   set(ht, 'Position', get(0,'Screensize'));
%   title([name(1:6) ': num(forest) = ' num2str(length(unique(forest)))]);
  export_fig([name '_forest_samples'], '-eps')
end