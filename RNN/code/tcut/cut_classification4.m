%% mutual information for p_connect

load ../../data/iccv09-allData-eval-140.mat
load '../../output/iccv09-allTrees-eval-140-backup.mat'
addpath ../tools/MIToolbox/

allMIs = cell(1,length(allData));
for j = 1:length(allData)
  
% acc =
% 
%     0.7567
% 
% 
% mpr =
% 
%     0.8412
% 	im = allData{j}.img;
%   fprintf('computing %d...\n', j);
%   load(['../../../BSR/iccv09data/ucm2/iccv09_' num2str(j) '.mat']);
%   ucm = ucm2(3:2:end, 3:2:end);
% 	
% 	regions = bwlabel(ucm == 0,4);
% 	edges = find(ucm > 0);
% 	[edgeI edgeJ] = ind2sub(size(ucm),edges);
% 	edgeStrength = ucm(edges);
% 	
% 	regions = padarray(regions,[1 1]);
% 	edges = sub2ind(size(regions), edgeI+1, edgeJ+1); %reflect the padding
% 	
% 	neighbors4 = [regions(edges-1) regions(edges+1) regions(edges+size(regions,1)) regions(edges-size(regions,1))];
% 	neighbors8 = [neighbors4 ...
% 	    regions(edges-1-size(regions,1)) regions(edges+1-size(regions,1)) ...
% 	    regions(edges-1+size(regions,1)) regions(edges+1+size(regions,1))];
% 	
% 	neighbors4 = neighbors4';
% 	neighbors8 = neighbors8';
% 	
% 	nRegions = max(regions(:));
% 	
% 	bLength4 = zeros(nRegions);
% 	bLength8 = zeros(nRegions);
% 	bStrength = zeros(nRegions);
% 	
% 	for i = 1:size(neighbors8,2)
% 	    t = unique(neighbors8(:,i));
% 	    bLength8(t(2:end),t(2:end))=bLength8(t(2:end),t(2:end))+1;
% 	    t = unique(neighbors4(:,i));
% 	    bLength4(t(2:end),t(2:end))=bLength4(t(2:end),t(2:end))+1;   
% 	end
% 	
% 	bLength8(bLength8==1) = 0; %removing "accross corner" neighbors 
% 	bLength = (bLength4+bLength8)*0.5; %reasonable approximation to the Euclidean length
% 	
% 	bLength = bLength-diag(diag(bLength));
% 	
% 	[nbr1 nbr2] = ind2sub(size(bLength),find(bLength));
% 	t = find(nbr1 > nbr2);
% 	nbr1(t) = [];
% 	nbr2(t) = [];
% 	
% 	where8 = cell(nRegions,1);
% 	
% 	for i = 1:nRegions
% 	    where8{i} = find(any(neighbors8 == i));
% 	end
% 	
% 	for i = 1:numel(nbr1)
% 	    bStrength(nbr1(i),nbr2(i)) = median(edgeStrength(intersect(where8{nbr1(i)},where8{nbr2(i)})));
% 	end
% 	bStrength = max(bStrength,bStrength');
% 	
% 	%filling in edge pixels
% 	el = strel('diamond',1); 
% 	for i = 1:2
% 	   tmp = imdilate(regions,el);
% 	   regions(regions == 0) = tmp(regions == 0);
% 	end
% 	
% 	bStrength(bStrength == 0) = +inf;
% 	bStrength(sub2ind(size(bStrength),1:size(bStrength,1),1:size(bStrength,1))) = 0;
%   bStrength = 1 - bStrength;

  % get p_connect_pred
  nRegions = length(allData{j}.segLabels);
  numLeafNodes = nRegions;
  numTotalNodes = nRegions + nRegions -1;
  leafsUnder = cell(numLeafNodes,1);
  p_connect_pred = zeros(numTotalNodes,1);

  for s = 1:numLeafNodes
      leafsUnder{s} = s;
  end
% 
%   for n = numLeafNodes+1:numTotalNodes
%       kids = allTrees{j}.getKids(n);
%       leafsUnder{n} = [leafsUnder{kids(1)} leafsUnder{kids(2)}];
% 
% 	  %[p1 p2] = meshgrid(leafsUnder{kids(1)}, leafsUnder{kids(2)});
% 	  %p12 = [p1(:) p2(:)];
% 	  %MIs = bStrength(p1,p2);
%     MIs = bStrength(leafsUnder{kids(1)}, leafsUnder{kids(2)});
%     
%     p_connect_pred(kids(1)) = max(max(MIs));
%     p_connect_pred(kids(2)) = max(max(MIs));
%   end

  for n = numLeafNodes+1:numTotalNodes
    kids = allTrees{j}.getKids(n);
    leafsUnder{n} = [leafsUnder{kids(1)} leafsUnder{kids(2)}];
    [p1 p2] = meshgrid(leafsUnder{kids(1)}, leafsUnder{kids(2)});
	  p12 = [p1(:) p2(:)];
    MIs = -1 .* ones(1,length(p1));
    for m = 1:length(p1)
      %MIs(m) = mi(allData{j}.expectSegLabels(kids(1),:), allData{j}.expectSegLabels(kids(2),:));
      %MIs(m) = pdist2(allData{j}.expectSegLabels(p1(m),:), allData{j}.expectSegLabels(p2(m),:),'correlation');
      MIs(m) = h([allData{j}.expectSegLabels(p1(m),:)', allData{j}.expectSegLabels(p2(m),:)']) - h(allData{j}.expectSegLabels(p1(m),:)');
      MIs(m) = 1 - MIs(m);
    end
    assert(all(MIs <= 1) && all(MIs >= 0));
    
    p_connect_pred(kids(1)) = max(MIs);
    p_connect_pred(kids(2)) = max(MIs);
  end

  allMIs{j} = p_connect_pred;
end
