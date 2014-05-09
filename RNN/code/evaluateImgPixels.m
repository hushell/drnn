function [acc mpr] = evaluateImgPixels(allData,clsf)
addpath ~/working/deep/netlab3_3/
Z = cell(1,length(allData));

if clsf >= 3 && clsf <= 4 % RNN
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
      
      load prior_with_position.mat % px
      %px = [0.1499 0.1398 0.2055 0.0927 0.0415 0.2420 0.0050 0.1237];
      nregions = size(px,1);
      til = nregions^(1/2);
      
      for i = 1:length(allTrees)
        if clsf == 3.5
          numLeafNodes = size(allData{i}.adj,1);
          % mask
          [iy,ix] = size(allData{i}.segs2);
          mask = zeros(iy,ix);
          for c = 1:til
            for b = 1:til
              if b ~= til
                mask((b-1)*ceil(iy/til)+1:b*ceil(iy/til), (c-1)*ceil(ix/til)+1:c*ceil(ix/til)) = b+(c-1)*til;
              else
                mask((b-1)*ceil(iy/til)+1:end, (c-1)*ceil(ix/til)+1:c*ceil(ix/til)) = b+(c-1)*til;
              end
            end
          end
          mask = mask(1:iy,1:ix);
          
          overlap = zeros(length(allData{i}.segLabels),nregions);
          for si = 1:length(allData{i}.segLabels)
            for oi = 1:nregions
              simg = allData{i}.segs2 == si;
              bimg = mask == oi;
              intimg = simg & bimg;
              overlap(si,oi) = sum(intimg(:)) / sum(bimg(:));
            end
          end
          px_sp = overlap*px;
          px_sp = bsxfun(@rdivide,px_sp,sum(px_sp,2));
          Z{i} = allTrees{i}.catOut(:,1:numLeafNodes)';
          Z{i} = Z{i} ./ px_sp;
        else
          Z{i} = allTrees{i}.catOut';
        end
      end
	end
elseif clsf == 0 % lr
	%load lr_119_iccv09.mat
  load lr_iccv09_140.mat
	for i = 1:length(allData)
	    if length(allData{i}.segLabels)~=size(allData{i}.feat2,1)
	        disp(['Image ' num2str(i) ' has faulty data, skipping!'])
	        continue
      end
      Z{i} = glmfwd(net, allData{i}.feat2);
  end
elseif clsf == 0.1
  load lr_iccv09_119_kernel.mat
	for i = 1:length(allData)
	    if length(allData{i}.segLabels)~=size(allData{i}.feat2,1)
	        disp(['Image ' num2str(i) ' has faulty data, skipping!'])
	        continue
	    end
		Z{i} = glmfwd(net, vl_homkermap(allData{i}.feat2',1)');
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
