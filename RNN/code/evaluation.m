function [cm, cmev, pcorr, pcorrev] = evaluation(class, classev, t, tev, y, yev, ...
    nclass, nsp, nspev, pixcsp, pixcspev)
% compute SP percent correct per class and confusion matrix

ctot = sum(t);  % number of SP per class
ctotev = sum(tev);

cm = zeros(nclass); % confusion matrix
cmev = zeros(nclass); % confusion matrix

%nsp = size(X,1);
for i = 1:nsp
  cm(y(i),class(i)) = cm(y(i),class(i)) +1;
end

%nspev = size(Xev,1);
for i = 1:nspev
  cmev(yev(i),classev(i)) = cmev(yev(i),classev(i)) +1;
end

pcorr = diag(cm) ./ ctot';
pcorrev = diag(cmev) ./ ctotev';

labels = {'Hair', 'Face', 'Upper Clothes', 'Lower Clothes', 'Arms', 'Legs', 'Background'};

fprintf('\n\n SEGMENTWISE ACCURACY \n\n', ' ', 'train', ' eval');
fprintf('%15s %5s %5s\n', ' ', 'train', ' eval');
for i = 1:nclass
  fprintf('%15s %5.3f %5.3f\n',labels{i}, pcorr(i), pcorrev(i));
end
fprintf('%15s %5.3f %5.3f\n', 'Average', mean(pcorr), mean(pcorrev));

% now do *pixelwise* correctness calculation

cmp = zeros(nclass); % confusion matrix pixelwise

%nsp = size(X,1);
for i = 1:nsp
  tmp = pixcsp(i,:);
  cmp(:,class(i)) =   cmp(:,class(i)) + tmp';
end

cmp2 = zeros(nclass);
for i = 1:nsp
  tmp = pixcsp(i,:);
  cmp2(y(i),:) =  cmp2(y(i),:) + tmp;
end

cmpev = zeros(nclass); % confusion matrix pixelwise

%nspev = size(Xev,1);
for i = 1:nspev
  tmp = pixcspev(i,:);
  cmpev(:,classev(i)) =   cmpev(:,classev(i)) + tmp';
end

cmpev2 = zeros(nclass);
for i = 1:nspev
  tmp = pixcspev(i,:);
  cmpev2(yev(i),:) =  cmpev2(yev(i),:) + tmp;
end

pcorrp = diag(cmp) ./ sum(cmp,2);
pcorrpev = diag(cmpev) ./ sum(cmpev,2);
pcorrp_ub = diag(cmp2) ./ sum(cmp2,2);
pcorrpev_ub = diag(cmpev2) ./ sum(cmpev2,2);

fprintf('\n\n PIXELWISE ACCURACY \n\n', ' ', 'train', ' eval');
fprintf('%15s %5s %5s\n', ' ', 'train', ' eval');
for i = 1:nclass
  fprintf('%15s %5.3f %5.3f\n',labels{i}, pcorrp(i), pcorrpev(i));
end
fprintf('%15s %5.3f %5.3f\n', 'Average', mean(pcorrp), mean(pcorrpev));

fprintf('\n\n PIXELWISE ACCURACY UPPER BOUND \n\n', ' ', 'train', ' eval');
fprintf('%16s %5s %5s\n', ' ', 'train', ' eval');
for i = 1:nclass
  fprintf('%15s % 5.3f %5.3f\n',labels{i}, pcorrp_ub(i), pcorrpev_ub(i));
end
fprintf('%15s % 5.3f %5.3f\n', 'Average', mean(pcorrp_ub), mean(pcorrpev_ub));
