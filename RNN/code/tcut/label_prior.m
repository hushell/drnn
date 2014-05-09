function px = label_prior()

addpath ../
load ../../data/iccv09-allData-eval.mat
run('~/working/deep/vlfeat-0.9.16/toolbox/vl_setup.m');

R = 16;
til = R^(1/2);
nclass = 8;
px = zeros(R,nclass) + 1e-10;

%masks = cell(1,length(allData));
for i = 1:length(allData)
  i
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
  %imagesc(mask);
  %masks{i} = mask;
  
  for b = 1:R
    counts = zeros(1,nclass);
    cmsk = mask == b;
    labimg = allData{i}.labels;
    for k = 1:nclass
      cimg = labimg(cmsk) == k;
      counts(k) = sum(cimg(:)) / sum(cmsk(:));
    end
    %assert(sum(counts) == 1); % cannot assert due to 0 in labels
    px(b,:) = px(b,:) + counts;
  end
end
%save masks_iccv09_eval.mat masks

px = bsxfun(@rdivide,px,sum(px,2));

