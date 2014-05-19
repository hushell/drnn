function [accs mprs] = test_CG_xvalid(data_train, data_eval, save_dir)
% logistic regression with 5-fold cross validation

if nargin < 1
    data_train = '../../data/iccv09-allData-train-140.mat';
    data_eval = '../../data/iccv09-allData-eval-140.mat';
    save_dir = './CG_iccv09-140';
end

%addpath ~/working/deep/netlab3_3/
%addpath /scratch/working/deep/netlab3_3/
addpath ../

accs = zeros(1,5);
mprs = zeros(1,5);

%% prepare data
load(data_train); 
[nr, nim] = size(allData);  % nr is 1, nim is the number of images
D = size(allData{1}.feat2,2); % number of features
nclass = 8;

if ~isfield(allData{1}, 'labelCountsPerSP')
    % numLabelsPerSP
    for i = 1:length(allData)
        labelRegs = allData{i}.labels;
        segs = allData{i}.segs2;
        numSegs = max(segs(:));
        labelCountsPerSP = zeros(numSegs, nclass);
        numPixelInSP = zeros(numSegs, 1);
        for r = 1:numSegs
            numPixelInSP(r) = numel(labelRegs(segs == r));
            for ci = 1:nclass
                labelCountsPerSP(r,ci) = sum(labelRegs(segs==r) == ci);
            end
        end
        allData{i}.labelCountsPerSP = labelCountsPerSP;
        allData{i}.numPixelInSP = numPixelInSP;
    end
    
    % expectation observation
    for i = 1:length(allData)
        labelRegs = allData{i}.labels;
        segs = allData{i}.segs2;
        numSegs = max(segs(:));
        expectSegLabels = zeros(numSegs, nclass);
        for r = 1:numSegs
            numPixelInSeg = numel(labelRegs(segs == r));
            for ci = 1:nclass
                expectSegLabels(r,ci) = sum(labelRegs(segs==r) == ci) / numPixelInSeg;
            end
        end
        allData{i}.expectSegLabels = expectSegLabels;
    end
    save ../../data/iccv09-allData-train.mat allData
end

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
	%[Xe te] = form_data(allDataFold{xv}, nsp_fold{xv}, D);
  allData = [allDataFold{setdiff(1:5,xv)}];
  [Xr tr yr] = form_data(allData, sum(nsp_fold)-nsp_fold(xv), D, nclass);
  clear allData

  net_file = [save_dir '_fold_' num2str(xv) '.mat'];
  if exist(net_file, 'file')
      load(net_file)
  else
	% counting grid model for each class
	cgm = cell(1,nclass);
	E = [30 30];
	W = [4 4];
	
	for k = 1:nclass
	  counts = Xr(yr == k, :)';
	  [pi,pl,Lq,loglikelihood_samples] = cg( counts, E, W);
	  cgm{k}.pi = pi;
	  cgm{k}.pl = pl;
	  cgm{k}.Lq = Lq;
	  cgm{k}.loglikelihood_samples = loglikelihood_samples;
	end
	save(net_file,'cgm');
  end

	fprintf('testing in fold %d...\n', xv);
	options.learn_pi = 0;
	options.learn_pl = 0;
	Z = cell(1,143);
	for i = 1:143
      if length(allDataFold{xv}{i}.segLabels)~=size(allDataFold{xv}{i}.feat2,1)
	    	disp(['Image ' num2str(i) ' has faulty data, skipping!'])
	    	continue
      end
		
  		Xi = allDataFold{xv}{i}.feat2;
  		Zk = zeros(nclass,size(Xi,1));
  		for k = 1:nclass
  		  options.pi = cgm{k}.pi;
  		  options.pl = cgm{k}.pl;
  		  [~,~,~,loglik] = cg(Xi', E, W, options);
  		  Zk(k,:) = loglik;
  		end
        Z{i} = Zk;
	end

	[accs(xv) mprs(xv)] = evaluateImgPixels2(allDataFold{xv}, Z);
	fprintf('fold %d: acc = %f, mpr = %\n', xv, accs(xv), mprs(xv));
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

