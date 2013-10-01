% subtree_check.m: check which subtrees correspond to parts 
% hair 1, face 2, upper 3, lower 4, arm 5, leg 6, bg 7

% load [W, Wbot, Wcat, Wout], params and allData
load ../output/penn-fudan-all_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05.mat
load ../data/penn-fudan-128-allData-train.mat

% compute all parse trees
tree_file = '../output/penn-fudan-allTrees-train.mat';
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
        save('../output/penn-fudan-allTrees-train.mat', 'allTrees');
    end
end

% test DP
make_change_points(allData{1}, allTrees{1}, 1);

bg = 7;
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

% visualizeParseTree3D(allData{61},Wbot,W,Wout,Wcat,params,-1,subtree_ids{61}{3});
% visualizeParseTree3D(allData{76},Wbot,W,Wout,Wcat,params,-1,subtree_ids{76}{4});
% visualizeParseTree3D(allData{3},Wbot,W,Wout,Wcat,params,-1,subtree_ids{3}{5});
% visualizeParseTree3D(allData{15},Wbot,W,Wout,Wcat,params,-1,subtree_ids{15}{6});