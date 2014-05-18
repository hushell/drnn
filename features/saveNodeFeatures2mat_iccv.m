%

startup_iccv09

for i = 1:length(allData)
    [numNodes, H] = readNodeFeatures(['iccv09_' num2str(i)], spfeat_dir);
    save([data_dir, '/feat_superpixels/', 'iccv09_', num2str(i), '_node.mat'], 'H');
    %[numNodes, H, numEdges, E, S] = readSPFeatures(['iccv09_' num2str(i)], spfeat_dir);
    %save([data_dir, '/feat_superpixels/', 'iccv09_', num2str(i), '_node_edge.mat'], 'H', 'E', 'S');
end
