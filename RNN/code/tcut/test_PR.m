% test PR
vis = 1;

addpath ../
load ../../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
load ../../data/iccv09-allData-eval.mat
run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');

% compute all parse trees
tree_file = '../../output/iccv09-allTrees-eval.mat';
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
        save('../../output/iccv09-allTrees-eval.mat', 'allTrees');
    end
end

theta_plus = MLE_theta(allData,8);

fp = fopen('equally-good.txt', 'w');
fprintf(fp, 'start at %d-%d-%d  %d:%d:%d\n', clock);
fclose(fp);

% for each image
for i = 4:length(allData)
    fp = fopen('equally-good.txt', 'a');
    fprintf(fp, '\n\n************ file %d ************\n\n', i);
    
    % get which p_connect will be used
    [~,~,~,~,q_max_diff,q_max_diff_ind] = tree_cut_postpone(allData{i}, allTrees{i}, theta_plus, 8, -1);
    fprintf(fp, '>>> q_max_diff: \n');
    fprintf(fp, '>>> '); fprintf(fp, '%f ', q_max_diff); fprintf(fp, '\n');
    fprintf(fp, '>>> '); fprintf(fp, '%f ', q_max_diff_ind); fprintf(fp, '\n\n');
    
    fclose(fp);
    
    p_samples = unique(q_max_diff);
    p_samples = 1./(9 - exp(p_samples));
    p_samples = [0, p_samples, 0.5, 1];
    p_samples = sort(p_samples(p_samples >= 0));
    p_samples = unique(p_samples);
    %p_samples = 0:0.05:1.0;
    
    [PRs,~,nCuts] = evalSegPerImg(allData{i}, allTrees{i}, theta_plus, 8, p_samples,1);
    [peak,loc] = max(PRs);
%     lb = max(0,p_samples(loc)-0.01);
%     ub = min(1,p_samples(loc)+0.01);
%     step_size = 0.001;
%     p_samples = lb:step_size:ub;
%     [PRs,~,nCuts] = evalSegPerImg(allData{i}, allTrees{i}, theta_plus, 8, p_samples,vis);
%     [peak,loc] = max(PRs);
    fprintf('>>>>>> %d best PR value = %f at %f\n', i, peak, p_samples(loc));
    fprintf('------ nCuts = %d\n', nCuts(loc));
    if (vis)
        pause
    end
end