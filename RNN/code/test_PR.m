% test PR

for i = 1:length(allData)
    evalSegPerImg(allData{i}, allTrees{i}, theta_plus, 8);
    pause
end