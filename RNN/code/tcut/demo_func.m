function demo_func(allData, allTrees, theta_plus, i)

[Q1,cuts1,labs1,forest1] = tree_cut_new(allData{i}, allTrees{i}, theta_plus, 8, 0.13, 0, 0);
[Q,cuts,labs,forest] = tree_cut_postpone(allData{i}, allTrees{i}, theta_plus, 8, 0.13, 0, 0);
plot_comparison(allTrees{i},labs,labs1,forest,forest1,i);

% helper function
function plot_comparison(imgTree,labs,labs1,forest,forest1,i)
figure(i*1000+1); imgTree.plotLabs(labs); title(['img ' num2str(i) ' : tree\_cut\_postpone : labels' ' : num(labels) = ' num2str(length(unique(labs)))]);
figure(i*1000+2); imgTree.plotLabs(labs1); title(['img ' num2str(i) ' : tree\_cut\_new : labels' ' : num(labels) = ' num2str(length(unique(labs1)))]);
figure(i*1000+3); imgTree.plotForest(forest); title(['img ' num2str(i) ' : tree\_cut\_postpone : forest' ' : num(forest) = ' num2str(length(unique(forest)))]);
figure(i*1000+4); imgTree.plotForest(forest1); title(['img ' num2str(i) ' : tree\_cut\_new : forest' ' : num(forest) = ' num2str(length(unique(forest1)))]);


