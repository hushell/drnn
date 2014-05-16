% SIFT features

if matlabpool('size') == 0
    matlabpool 4;
end

nWords = 512;
nFeatures = nWords * 100;
nImgs = length(allData);

fprintf('Computing SIFT features\n') ;
descriptors = {};
cdescriptors = cell(1,numel(names));
for i = 1:length(allData)
  im = im2single(allData{i}.img) ;
  [f, d] = vl_phow(im, 'step', 1, 'sizes', 8, 'floatdescriptors', true) ;
  cdescriptors{i} = vl_colsubset(d, round(nFeatures / nImgs), 'uniform') ;
end
descriptors = cat(2, descriptors, cdescriptors);

if matlabpool('size') > 0
    matlabpool close;
end

fprintf('Computing visual words and kdtree\n') ;
tic;
descriptors = single([descriptors{:}]) ;
vocabulary.words = vl_kmeans(descriptors, nWords, 'verbose', 'algorithm', 'elkan') ;
vocabulary.kdtree = vl_kdtreebuild(vocabulary.words) ;
fprintf('TIME in building vocabulary is %f', toc);
save vocabulary.mat vocabulary


counts = zeros(nWords,nImgs*1000);
cnt = 1;
for i = 1:length(allData)
  im = im2single(allData{i}.img) ;
  [f, d] = vl_phow(im, 'step', 1, 'sizes', 8, 'floatdescriptors', true) ;
  counts(:,cnt) = quantizeDescriptors(vocabulary, d) ;
end

save counts_iccv09.mat counts