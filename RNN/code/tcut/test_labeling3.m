% counting grid training
addpath ..
load ../../data/iccv09-allData-eval-140.mat
% compute all parse trees
tree_file = '../../output/iccv09-allTrees-eval-140.mat';
if ~exist('allTrees','var')
    if exist(tree_file,'file')
        load(tree_file); 
    else
        allTrees = cell(1,length(allData));
        %allThres = cell(1,length(allData));
        for i = 1:length(allData)
            fprintf('computing %d...\n', i);
            load(['../../../BSR/iccv09data/ucm2/iccv09_' num2str(i) '.mat']);
            [thisTree,thres_arr] = buildBSRTree2(ucm2,0);
            allTrees{i} = thisTree;
            %allThres{i} = thres_arr;
        end
        test_labeling2;
        save('../../output/iccv09-allTrees-eval-140.mat', 'allTrees');
    end
end

[nr, nim] = size(allData);  % nr is 1, nim is the number of images
D = size(allData{1}.feat2,2); % number of features
nclass = 8;

nsp = 0; % number of superpixels
for i = 1:nim
  nsp = nsp +  size(allData{i}.segLabels,1);
end

X = zeros(nsp,D);  % data matrix
y = zeros(nsp,1);  % holds labels
count = 1;  % counter for each row of X
for i = 1:nim
     spi = size(allData{i}.feat2,1);  % number of SP in this image
     X(count:count+spi-1,:) =  allData{i}.feat2;
     y(count:count+spi-1,:) =  allData{i}.segLabels;
     count = count + spi;
end

% counting grid model for each class
cgm = cell(1,nclass);
E = [30 30];
W = [4 4];

for k = 1:nclass
  counts = X(y == k, :)';
  [pi,pl,Lq,loglikelihood_samples] = cg( counts, E, W);
  cgm{k}.pi = pi;
  cgm{k}.pl = pl;
  cgm{k}.Lq = Lq;
  cgm{k}.loglikelihood_samples = loglikelihood_samples;
end

options.learn_pi = 0;
options.learn_pl = 0;
for i = 1:nim
  Xi = allData{i}.feat2;
  Z = zeros(nclass,size(Xi,1));
  
  for k = 1:nclass
    options.pi = cgm{k}.pi;
    options.pl = cgm{k}.pl;
    [~,~,~,loglik] = cg(Xi', E, W, options);
    Z(k,:) = loglik;
  end
  
  allTrees{i}.catOut = Z;
end
