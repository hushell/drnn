function [acc] = evaluateImgPixels(allData,clsf)

Z = cell(1,length(allData));

if clsf == 3 % RNN
  load ../../output/iccv09-1_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05_good.mat
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
	        save(tree_file, 'allTrees');
      end
      for i = 1:length(allTrees)
        Z{i} = allTrees{i}.catOut';
      end
	end
elseif clsf == 0 % lr
	load lr_119_iccv09.mat
	for i = 1:length(allData)
	    if length(allData{i}.segLabels)~=size(allData{i}.feat2,1)
	        disp(['Image ' num2str(i) ' has faulty data, skipping!'])
	        continue
	    end
		Z{i} = glmfwd(net, allData{i}.feat2);
	end
elseif clsf == 2 % gmm
	load gmm_119_iccv09.mat
  nclass = length(nets);
  for i = 1:length(allData) 
    Z{i} = zeros(length(allData{i}.segLabels),nclass);
    for k = 1:nclass
      mix = nets{k};
      Z{i}(:,k) = gmmprob(mix, allData{i}.feat2);
    end
  end
elseif clsf == 1
  ;
elseif clsf == -1 % tcut
  tree_file = '../../output/iccv09-allTrees-eval.mat';
  load(tree_file);
  % to get p_connect_star
  if ~exist('p_connect_star', 'var')
    if exist('p_connect_star.mat', 'file')
      load p_connect_star.mat
    else
      test_PR_new;
    end
  end

  theta_plus = MLE_theta(allData,8);
  
  for i = 1:length(allData)
    [~,~,Z{i},~] = tree_cut_new(allData{i}, allTrees{i}, theta_plus, 8, p_connect_star(i));
  end
end


allCorrectPixels = 0;
allPixels = 0;
for i = 1:length(allData)
    if length(allData{i}.segLabels)~=size(allData{i}.feat2,1)
        disp(['Image ' num2str(i) ' has faulty data, skipping!'])
        continue
    end
    [correctPixels totalPixelsImg] = evaluateOneImgPixels(allData{i},Z{i});
    allCorrectPixels = allCorrectPixels + correctPixels ;
    allPixels = allPixels + totalPixelsImg;
    if mod(i,10)==0
        disp(['Done with computing image ' num2str(i)]);
    end
end
acc = allCorrectPixels/allPixels
