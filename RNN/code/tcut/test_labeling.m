% labeling tests
clear
addpath ../
%load ../../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
%load ../../data/iccv09-allData-eval.mat
run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');
addpath ~/working/deep/netlab3_3/
%theta_plus = MLE_theta(allData,8);

load ~/working/deep/chris_netlab_code/dat119_iccv09.mat

%% training
classifier = 2;
nclass = 8;
D = size(X,2);

if classifier == 2
  ncentres = 1;
  input_dim = D;

  nets = cell(1,nclass);
  for i = 1:nclass
    mixinit = gmm(input_dim, ncentres, 'diag');
    select = y == i;
    data = X(select,:);
    options = foptions;
    options(14) = 5;	% Just use 5 iterations of k-means in initialisation
    % Initialise the model parameters from the data
    mix = gmminit(mixinit, data, options);

    % Set up vector of options for EM trainer
    options = zeros(1, 18);
    options(1)  = 1;		% Prints out error values.
    options(14) = 20;		% Number of iterations.

    [mix, options, errlog] = gmmem(mix, data, options);
    nets{i} = mix;
    save gmm_119_iccv09.mat nets
  end
elseif classifier == 0
  net = glm(D, nclass, 'softmax');
  options = foptions;
  options(1) = 1; % set to 1 to display error values during training
  options(14) = 10; %  maximum number of iterations 
  net = glmtrain(net, options, X, t);
elseif classifier == 0.5
  X = vl_homkermap(X',1);
  X = X';
  D = size(X,2);
  net = glm(D, nclass, 'softmax');
  options = foptions;
  options(1) = 1; % set to 1 to display error values during training
  options(14) = 10; %  maximum number of iterations 
  net = glmtrain(net, options, X, t);
end

%% testing
return
% load ~/working/deep/chris_netlab_code/chris_net/lr_119_iccv.mat
% load ~/working/deep/chris_netlab_code/chris_net/mlp_119_iccv09_50_2000.mat
% load ~/working/deep/chris_netlab_code/chris_net/mlp_119_iccv09_50_4000.mat

if classifier == 0 %'LR'
  Z = glmfwd(net, X);
  [foo, class] = max(Z');
  Zev = glmfwd(net, Xev);
  [fooev, classev] = max(Zev');
elseif classifier == 1 %'MLP'
  Z = mlpfwd(net, X);
  [foo, class] = max(Z');
  Zev = mlpfwd(net, Xev);
  [fooev, classev] = max(Zev');
elseif classifier == 2 %'GMM'
  Z = zeros(size(y),nclass);
  Zev = zeros(size(yev),nclass);
  for i = 1:nclass
    mix = nets{i};
    Z(:,i) = gmmprob(mix, X);
    Zev(:,i) = gmmprob(mix, Xev);
  end
  [foo, class] = max(Z');
  [fooev, classev] = max(Zev');
end


% compute SP percent correct per class and confusion matrix

ctot = sum(t);  % number of SP per class
ctotev = sum(tev);

cm = zeros(nclass); % confusion matrix
cmev = zeros(nclass); % confusion matrix

nsp = size(X,1);
for i = 1:nsp
  cm(y(i),class(i)) = cm(y(i),class(i)) +1;
end

nspev = size(Xev,1);
for i = 1:nspev
  cmev(yev(i),classev(i)) = cmev(yev(i),classev(i)) +1;
end

pcorr = diag(cm) ./ ctot';
pcorrev = diag(cmev) ./ ctotev';

% labels = {'Hair', 'Face', 'Upper Clothes', 'Lower Clothes', 'Arms', 'Legs', 'Background'};
labels = {'sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain', 'fg'};

fprintf('\n\n SEGMENTWISE ACCURACY \n\n', ' ', 'train', ' eval');
fprintf('%15s %5s %5s\n', ' ', 'train', ' eval');
for i = 1:nclass
  fprintf('%15s %5.3f %5.3f\n',labels{i}, pcorr(i), pcorrev(i));
end
fprintf('%15s %5.3f %5.3f\n', 'Average', mean(pcorr), mean(pcorrev));


