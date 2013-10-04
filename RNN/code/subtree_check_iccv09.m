% subtree_check.m: check which subtrees correspond to parts 
%iccv09: 0 void   1,1 sky  0,2 tree   2,3 road  1,4 grass  1,5 water  1,6 building  2,7 mountain 2,8 foreground

% load [W, Wbot, Wcat, Wout], params and allData
load ../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
%load ../data/iccv09-allData-eval.mat

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

%% test DP
Q_all = cell(1,length(allData));
pix_err_all = cell(1,length(allData));
sp_err_all = cell(1,length(allData));
n_cuts_all = cell(1,length(allData));
samples = 0:10:100;
for i = 1:length(allData)
    fprintf('--- %d ---', i);
    [Q_all{i}, sp_err_all{i}, n_cuts_all{i}] = test_DP(allData{i}, allTrees{i}, 8, samples);
    n_pixels = size(allData{i}.img,1)*size(allData{i}.img,2);
    pix_err_all{i} = Q_all{i} ./ n_pixels;
end

save iccv09_dp.mat Q_all pix_err_all sp_err_all n_cuts_all

pix_err_mean = mean(reshape(cell2mat(pix_err_all), [size(pix_err_all{1},2), length(allData)]),2)';
sp_err_mean = mean(reshape(cell2mat(sp_err_all), [size(sp_err_all{1},2), length(allData)]),2)';
n_cuts_mean = mean(reshape(cell2mat(n_cuts_all), [size(n_cuts_all{1},2), length(allData)]),2)';

figure;
subplot(1,3,1); plot(samples, pix_err_mean, '-go', 'linewidth', 2); legend('pixel level loss');
subplot(1,3,2); plot(samples, sp_err_mean, '-bo', 'linewidth', 2); legend('superpixel level loss');
subplot(1,3,3); plot(samples, n_cuts_mean, '-ro', 'linewidth', 2); legend('number of cuts');

%% check pure subtrees 
if 0
bg = 8;
subtree_ids = cell(1,length(allData));
labelsUnder = cell(1,length(allData));
purities = cell(1,length(allData));
for i = 1:length(allData)
    i
    [subtree_ids{i}, labelsUnder{i}, purities{i}] = subtree_check_single(allTrees{i},allData{i},-1,bg);
end

n_parts = max(allData{1}.segLabels);
average_purities = zeros(1,n_parts);
max_purities = zeros(1,n_parts);
max_purities_loc = zeros(1,n_parts);
counts = zeros(1,n_parts); counts(bg) = 1;
for i = 1:length(allData)
    % TODO: handle some labels are lack
    if n_parts ~= max(allData{i}.segLabels)
        continue
    end
    for j = 1:n_parts
        if j == bg
            continue
        end
        if purities{i}{j} == -1
            continue
        else
            counts(j) = counts(j) + 1;
            average_purities(j) = average_purities(j) + purities{i}{j};
            if purities{i}{j} > max_purities(j)
                max_purities(j) = purities{i}{j};
                max_purities_loc(j) = i;
            end
        end
    end
end

average_purities = average_purities ./ counts
max_purities
max_purities_loc

% visualizeParseTree3D(allData{12},Wbot,W,Wout,Wcat,params,-1,subtree_ids{12}{1});
% visualizeParseTree3D(allData{48},Wbot,W,Wout,Wcat,params,-1,subtree_ids{48}{2});
% visualizeParseTree3D(allData{1},Wbot,W,Wout,Wcat,params,-1,subtree_ids{1}{3});
% visualizeParseTree3D(allData{69},Wbot,W,Wout,Wcat,params,-1,subtree_ids{69}{4});
% visualizeParseTree3D(allData{29},Wbot,W,Wout,Wcat,params,-1,subtree_ids{29}{5});
% visualizeParseTree3D(allData{135},Wbot,W,Wout,Wcat,params,-1,subtree_ids{135}{6});
% visualizeParseTree3D(allData{73},Wbot,W,Wout,Wcat,params,-1,subtree_ids{73}{7});
end