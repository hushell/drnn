function [accs mprs] = test_LR_xvalid()
% logistic regression with 5-fold cross validation
addpath ~/working/deep/netlab3_3/

accs = zeros(1,5);
mprs = zeros(1,5);

%% prepare data
load ../../data/iccv09-allData-train.mat
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

load ../../data/iccv09-allData-eval.mat
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
	[Xr tr] = form_data(allData, sum(nsp_fold)-nsp_fold(xv), D, nclass);
  clear allData

	net = glm(D, nclass, 'softmax');
	options = foptions;
	options(1) = 1; % set to 1 to display error values during training
	options(14) = 10; %  maximum number of iterations 
	nets{xv} = glmtrain(net, options, Xr, tr);
	save(['LR_iccv09_fold_' num2str(xv) '.mat'],'net');

	fprintf('testing in fold %d...\n', xv);
	Z = cell(1,143);
	for i = 1:143
      if length(allDataFold{xv}{i}.segLabels)~=size(allDataFold{xv}{i}.feat2,1)
	    	disp(['Image ' num2str(i) ' has faulty data, skipping!'])
	    	continue
      end
		
    	Z{i} = glmfwd(net, allDataFold{xv}{i}.feat2);
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
indx = y > 0;
y = y(indx);
X = X(indx,:);

% recode y in one-of-nclass format
id = eye(nclass);
t = id(y,:);

% % getting pixel counts per superpixel
% pixcsp = zeros(nsp,nclass);  % numper of pixels of each class per SP
% count = 1;  % counter for each row of pixcsp
% for i = 1:nim
%   seg = allData{i}.segs2;
%   esl = allData{i}.expectSegLabels;
%   seglin = seg(:);
%   nseg = max(seglin);
%   idsp = eye(nseg);
%   spsize =  sum(idsp(seglin,:));  % row vector
%   repspsize = (spsize')*ones(1,nclass);  % holds pixels per class per SP
%   pixcsp(count:count+nseg-1,:) = repspsize.*esl;
%   count = count + nseg;
% end
