function [accs mprs] = test_TC_xvalid(data_train, data_eval, save_dir, use_gt)
% logistic regression with 5-fold cross validation

if nargin < 1
    data_train = '../../data/iccv09-allData-train.mat';
    data_eval = '../../data/iccv09-allData-eval.mat';
    save_dir = './LR_iccv09-140';
    use_gt = 0;
end

%addpath ~/working/deep/netlab3_3/
%addpath /scratch/eecs-share/huxu/deep/netlab3_3/
addpath ../

accs = zeros(1,5);
mprs = zeros(1,5);

%% prepare data
fprintf('prepare data...\n');
load(data_train); 
[nr, nim] = size(allData);  % nr is 1, nim is the number of images
D = size(allData{1}.feat2,2); % number of features
nclass = 8;

allDataFold = cell(1,5);
for xv = 1:4
	allDataFold{xv} = allData(1,(xv-1)*143+1:xv*143);
end

nsp_fold = [];
nsp = 0; % number of superpixels
cnt = 0;
for i = 1:nim
  nsp = nsp +  size(allData{i}.segLabels,1);
  cnt = cnt + 1;
  if cnt >= 143
	  nsp_fold(end+1) = nsp;
	  cnt = 0;
	  nsp = 0;
  end
end
clear allData 

load(data_eval); 
allDataFold{5} = allData;

nsp = 0;
for i = 1:143
	nsp = nsp + size(allData{i}.segLabels,1);
end
nsp_fold(end+1) = nsp;
clear allData 

%% training & testing
nets = cell(1,5);
for xv = 1:5
  fprintf('training in fold %d...\n', xv);

  %net_file = [save_dir '_fold_' num2str(xv) '.mat'];
  net_file = '../../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat';
  if exist(net_file, 'file') && ~use_gt
      load(net_file)
  elseif use_gt
      fprintf('Use GT counts!\n');
      load(net_file)
  else
      error('RNN model has not been learned!\n');
  end

  allData = allDataFold{xv};

  % compute all parse trees
  tree_file = ['../../output/iccv09-allTrees-119-fold' num2str(xv) '.mat'];
  if ~exist('allTrees','var')
      if exist(tree_file,'file')
          load(tree_file); 
      else
          allTrees = cell(1,length(allData));
          for i = 1:length(allData)
              if length(allData{i}.segLabels)~=size(allData{i}.feat2,1)
                  disp(['Image ' num2str(i) ' has faulty data, skipping!'])
                  continue
              end
              topCorr=0;
              imgTreeTop = parseImage(topCorr,Wbot,W,Wout,Wcat,allData{i}.adj, ...
                  allData{i}.feat2,allData{i}.segLabels,params);
              allTrees{i} = imgTreeTop;
          end
          save(tree_file, 'allTrees');
      end
  end

  fprintf('computing p_connect_best... \n');
  theta_plus = MLE_theta(allData,8);

  p_connect_file = ['p_connect_star_119' 'fold_' num2str(xv) '.mat'];
  if exist(p_connect_file, 'file')
      load(p_connect_file);
  else
    p_connect_star = zeros(1,length(allData));
    for i = 1:length(allData) 
      if any(sum(allData{i}.labelCountsPerSP,2) == 0) ...
          || numel(unique(unique(allData{i}.segs2))) ~= length(allData{i}.segLabels) ...
          || length(allData{i}.segLabels) == 1
        p_connect_star(i) = -1;
        continue
      end
      
      % get which p_connect will be used
      [~,~,~,~,q_max_diff,q_max_diff_ind] = tree_cut_new(allData{i}, allTrees{i}, theta_plus, 8, -1);
      
      p_samples = unique(q_max_diff);
      p_samples = 1./(1+8./exp(p_samples)); % 1 / ( 1 + ( n/exp(q_diff) ) )
      assert(all(p_samples >= 0 & p_samples <= 1));
      p_samples = sort(p_samples(p_samples > 0));
      p_samples = [linspace(0,p_samples(1),10), linspace(p_samples(1),1,5), p_samples, 1/9];
      p_samples = unique(p_samples);
      p_samples = sort(p_samples);
      
      name = [];
      [PRs,spAccs,nCuts,PRs3,GCEs,VIs] = evalSegPerImg(name, @tree_cut_new, allData{i}, allTrees{i}, theta_plus, 8, p_samples);
      combo = (PRs.*spAccs.*PRs3) ./ (nCuts.*(VIs+1e-5));
      %[peak,loc] = max(PRs);
      [peak,loc] = fullmax(combo);
      [mnc,mmi] = min(nCuts(loc));
      p_connect_star(i) = p_samples(loc(mmi));

      fprintf('- image %d: best combo value = %f at %f\n', i, peak(mmi), p_samples(loc(mmi)));
      fprintf('- forest: number of subtrees = %d\n', mnc);
    end
    save(p_connect_file, 'p_connect_star')
  end


  fprintf('\ntesting in fold %d...\n', xv);
  Z = cell(1,143);
  for i = 1:143
    if length(allDataFold{xv}{i}.segLabels)~=size(allDataFold{xv}{i}.feat2,1) || numel(unique(unique(allData{i}.segs2))) ~= length(allData{i}.segLabels) || length(allData{i}.segLabels) == 1
      disp(['Image ' num2str(i) ' has faulty data, skipping!'])
      continue
    end
        	
    if use_gt %NOTE: change lik_type in tree_cut_new to 1
      [Q,cuts,Z{i},forest] = tree_cut_new(allData{i}, allTrees{i}, theta_plus, nclass, p_connect_star(i));
      %fprintf('>DEBUG Z{i} size = %d, %d\n', size(Z{i},1), size(Z{i},2));
      Z{i} = Z{i}';
    else
  	  %Zk = glmfwd(net, allDataFold{xv}{i}.feat2);
      %allTrees{i}.catOut = Zk';
      [~,~,Z{i},~] = tree_cut_new(allData{i}, allTrees{i}, theta_plus, nclass, p_connect_star(i));
      %fprintf('>DEBUG Z{i} size = %d, %d\n', size(Z{i},1), size(Z{i},2));
      Z{i} = Z{i}';
    end
  end

  [accs(xv) mprs(xv)] = evaluateImgPixels2(allDataFold{xv}, Z);
  fprintf('fold %d: acc = %f, mpr = %f\n', xv, accs(xv), mprs(xv));
  
  clear allTrees
end


%% helper functions
function [X t y] = form_data(allData, nsp, D, nclass)

nim = length(allData);
X = zeros(nsp,D);  % data matrix
y = zeros(nsp,1);  % holds labels

count = 1;  % counter for each row of X
for i = 1:nim
     spi = size(allData{i}.feat2,1);  % number of SP in this image
     X(count:count+spi-1,:) =  allData{i}.feat2;
     y(count:count+spi-1,:) =  allData{i}.segLabels;
     count = count + spi;
end
assert(max(y) == nclass);

% remove y = 0
indx = y ~= 0;
y = y(indx);
X = X(indx,:);

% recode y in one-of-nclass format
id = eye(nclass);
t = id(y,:);
