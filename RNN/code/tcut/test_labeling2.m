% create catOut for allTrees

classifier  = 0;
if classifier == 0 %'LR'
  load lr_iccv09_140.mat
  %Z = glmfwd(net, X);
  %[foo, class] = max(Z');
  %Zev = glmfwd(net, Xev);
  %[fooev, classev] = max(Zev');

  for i = 1:length(allData)
    if length(allData{i}.segLabels)~=size(allData{i}.feat2,1)
      disp(['Image ' num2str(i) ' has faulty data, skipping!'])
	    continue
    end
	X = allData{i}.feat2;	
  	Z = glmfwd(net, X);
  	[foo, clsl] = max(Z');
	allTrees{i}.catOut = Z';
	allTrees{i}.nodeLabels = clsl;
  end
end
