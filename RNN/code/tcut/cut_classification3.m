% tree_cut sampling
% needs to run cut_classification2 first
vis = 1;
stree = 1;

PR_peaks_pred = zeros(1,length(allData));
SPAccsMax_pred = zeros(1,length(allData));
%p_connect_star = zeros(1,length(allData));
for i = 1:length(allData)
    if vis && stree
      name = ['iccv09_forest_samples/iccv09_' num2str(i) '_'];
    else
      name = [];
    end
    
    evalSegPerImg4(name,@tree_cut_new,allData{i},allTrees{i},theta_plus,8,allppred{i},vis,stree);
    
    if vis && isempty(name)
        pause
    end
end