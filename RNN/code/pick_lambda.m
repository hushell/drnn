% pick_lambda
 
clear
close all


% load [W, Wbot, Wcat, Wout], params and allData
load ../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
load ../data/iccv09-allData-eval.mat

% compute all parse trees
tree_file = '../output/iccv09-allTrees-eval.mat';
if ~exist('allTrees','var')
    if exist(tree_file,'file')
        load(tree_file); 
    else
        allTrees = cell(1,length(allData));
        parfor i = 1:length(allData)
            if length(allData{i}.segLabels)~=size(allData{i}.feat2,1)
                disp(['Image ' num2str(i) ' has faulty data, skipping!'])
                continue
            end
            topCorr=0;
            imgTreeTop = parseImage(topCorr,Wbot,W,Wout,Wcat,allData{i}.adj, ...
                allData{i}.feat2,allData{i}.segLabels,params);
            allTrees{i} = imgTreeTop;
        end
        save('../output/iccv09-allTrees-eval.mat', 'allTrees');
    end
end

theta_plus = MLE_theta(allData,8);


n_labs = 8;
%lamb_samples = [0:0.05:1];
%lamb_samples = [0,0.00001,0.0001,0.001,0.01,0.1, 1, 10, 100, 1000, 10000, 100000];
%lamb_samples = [0.11111 0.111112 0.111114 0.111116 0.111118 0.11112];
lamb_samples = [0.01:0.04:0.21];

% number of images in the plotting, range from 1 to length(allData)
num_img = length(allData);


num_cc = zeros(1,num_img);
for i = 1:num_img
    for k = 1:n_labs
        [~,num] = bwlabel(allData{i}.labels == k);
        num_cc(i) = num_cc(i) + num;
    end
end

num_segs = zeros(length(lamb_samples),num_img);
j = 0;
for lambda = lamb_samples
    j = j + 1
    for i = 1:num_img
        [Q,cuts,labels] = tree_cut(allData{i}, allTrees{i}, theta_plus, n_labs, lambda);
        num_segs(j,i) = sum(cuts);
    end
    
    if mod(j,6) == 0
        figure(j);
        subplot(2,3,1); scatter(num_cc, num_segs(j-5,:), 'fill'); 
        line([0,max(num_cc)],[0,max(num_cc)], 'color', 'g', 'linestyle', '--', 'linewidth',2);
        %axis square;
        xlabel('num of connected comp in GT');
        ylabel('num of segments in forest');
        title(sprintf('lambda = %f', lamb_samples(j-5)));
        subplot(2,3,2); scatter(num_cc, num_segs(j-4,:), 'fill');
        line([0,max(num_cc)],[0,max(num_cc)], 'color', 'g', 'linestyle', '--', 'linewidth',2);
        %axis square;
        xlabel('num of connected comp in GT');
        ylabel('num of segments in forest');
        title(sprintf('lambda = %f', lamb_samples(j-4)));
        subplot(2,3,3); scatter(num_cc, num_segs(j-3,:), 'fill');
        line([0,max(num_cc)],[0,max(num_cc)], 'color', 'g', 'linestyle', '--', 'linewidth',2);
        %axis square;
        xlabel('num of connected comp in GT');
        ylabel('num of segments in forest');
        title(sprintf('lambda = %f', lamb_samples(j-3)));
        subplot(2,3,4); scatter(num_cc, num_segs(j-2,:), 'fill');
        line([0,max(num_cc)],[0,max(num_cc)], 'color', 'g', 'linestyle', '--', 'linewidth',2);
        %axis square;
        xlabel('num of connected comp in GT');
        ylabel('num of segments in forest');
        title(sprintf('lambda = %f', lamb_samples(j-2)));
        subplot(2,3,5); scatter(num_cc, num_segs(j-1,:), 'fill');
        line([0,max(num_cc)],[0,max(num_cc)], 'color', 'g', 'linestyle', '--', 'linewidth',2);
        %axis square;
        xlabel('num of connected comp in GT');
        ylabel('num of segments in forest');
        title(sprintf('lambda = %f', lamb_samples(j-1)));
        subplot(2,3,6); scatter(num_cc, num_segs(j-0,:), 'fill');
        line([0,max(num_cc)],[0,max(num_cc)], 'color', 'g', 'linestyle', '--', 'linewidth',2);
        %axis square;
        xlabel('num of connected comp in GT');
        ylabel('num of segments in forest');
        title(sprintf('lambda = %f', lamb_samples(j-0)));
    end
end
