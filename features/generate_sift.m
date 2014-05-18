% SIFT features

load ../RNN/data/iccv09-allData-train.mat

nWords = 512;
nFeatures = nWords * 5;
nImgs = length(allData);

if exist('vocabulary.mat','file')
  load vocabulary.mat
else
  fprintf('Computing SIFT features\n') ;
  descriptors = {};
  cdescriptors = cell(1,nImgs);
  for i = 1:length(allData)
    im = im2single(allData{i}.img) ;
    [f, d] = vl_phow(im, 'step', 1, 'sizes', 8, 'floatdescriptors', true) ;
    cdescriptors{i} = vl_colsubset(d, nFeatures, 'uniform') ;
  end
  descriptors = cat(2, descriptors, cdescriptors);
  
  fprintf('Computing visual words and kdtree\n') ;
  tic;
  descriptors = single([descriptors{:}]) ;
  vocabulary.words = vl_kmeans(descriptors, nWords, 'verbose', 'algorithm', 'elkan') ;
  vocabulary.kdtree = vl_kdtreebuild(vocabulary.words) ;
  fprintf('TIME in building vocabulary is %f', toc);
  save vocabulary.mat vocabulary
end

sift_dir = 'data_iccv09/parts_sift/';
%counts = zeros(nWords,nImgs*1000);
cnt = 1;
for i = 1:length(allData)
  im = im2single(allData{i}.img) ;
  im = padarray(im, [12 12]);
  [f, d] = vl_phow(im, 'step', 1, 'sizes', 8, 'floatdescriptors', true) ;
  %counts(:,cnt) = quantizeDescriptors(vocabulary, d) ;
  [words,~] = vl_kdtreequery(vocabulary.kdtree, vocabulary.words, ...
                                   d, 'MaxComparisons', 15) ;
  %words = double(words) ;
  tmfn = sprintf('%s/%s_sift.dat', sift_dir, ['iccv09_' num2str(i)]);
  fidtm = fopen(tmfn, 'w');
  
  fprintf(fidtm, '%d\n', words');
  fclose(fidtm);
end

%save counts_iccv09.mat counts
