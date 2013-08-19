%

%startup_directory;
startup_penn_fudan;

files = dir(lfw_dir);
images = cell(length(files)-2,1);
for j = 3:length(files)
    images{j-2} = files(j).name;
end

X(numel(images)) = struct();
for i = 1:numel(images)
    [numNodes, H] = readNodeFeatures(images{i}(1:end-4), spfeat_dir);
    save([data_dir, '/feat_superpixels/', images{i}(1:end-4), '_node_context.mat'], 'H');
end
