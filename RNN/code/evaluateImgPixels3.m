function [acc mpr] = evaluateImgPixels3(allData,allTrees,allppred)
addpath ~/working/deep/netlab3_3/
Z = cell(1,length(allData));

theta_plus = MLE_theta(allData,8);

%clsl = 2;
if nargin < 3
  if ~exist('p_connect_star', 'var')
    if exist('p_connect_star_140.mat', 'file')
      load p_connect_star_140.mat
    else
      test_PR_new_iccv09;
    end
  end
  
  % make sure allData and allTrees are for BSR trees
  for i = 1:length(allData)
    [~,~,Z{i},~] = tree_cut_new(allData{i}, allTrees{i}, theta_plus, 8, p_connect_star(i));
  end
else
  for i = 1:length(allData)
    [~,~,Z{i},~] = tree_cut_new(allData{i}, allTrees{i}, theta_plus, 8, allppred{i});
  end
end


allPR = zeros(1,length(allData));
allCorrectPixels = 0;
allPixels = 0;
for i = 1:length(allData)
    if length(allData{i}.segLabels)~=size(allData{i}.feat2,1)
        disp(['Image ' num2str(i) ' has faulty data, skipping!'])
        continue
    end
    [correctPixels totalPixelsImg allPR(i)] = evaluateOneImgPixels(allData{i},Z{i});
    allCorrectPixels = allCorrectPixels + correctPixels ;
    allPixels = allPixels + totalPixelsImg;
    if mod(i,10)==0
        disp(['Done with computing image ' num2str(i)]);
    end
end
acc = allCorrectPixels/allPixels;
mpr = mean(allPR);
