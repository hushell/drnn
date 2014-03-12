% extract features for superpixels by averaging and normalizing features of
% each pixel inside the superpixel
clear
close all
addpath(genpath('.'))
run('/home/hushell/working/deep/vlfeat-0.9.16/toolbox/vl_setup');
randn('state',0) ;
rand('state',0) ;
% figure(1) ; clf ;

rt_path = 'data_weizmann_horse/Color/';
save_path = 'data_weizmann_horse/feat_superpixels/';
files = dir(rt_path);
images = cell(length(files)-2,1);
for j = 3:length(files)
    images{j-2} = files(j).name;
end

% % prepare list.txt
% fp = fopen('list.txt', 'wb');
% for i = 1:numel(images)
% [hh, ww, ~] = size(imread([rt_path, images{i}])); 
% fprintf(fp, '%s %d %d\n', images{i}(1:end-4), hh, ww);
% end
% fclose(fp);

regionSize = 15;
regularizer = 0.3;
display = 0;
feat_params = {{'patches',0},{'position',1},{'fourier',1},{'hog',8}};
fdim = 100;

for id = 1:length(images)
    imfile = [rt_path, images{id}];
    im = double(imread(imfile)) / 255;
    
    % features for each pixel
    %pixelfeats = featurize_im(im,feat_params);
    
    % superpixels
    im = im2single(im) ;
    %L = vl_quickseg(im, 0.5, 2, 10);
    segments = vl_slic(im, regionSize, regularizer) ;
    %save([save_path, images{id}, '_seg.mat'], 'segments');
    %imwrite(label2rgb(segments), [save_path, images{id}(1:end-4), '_seg.ppm']);
    
    %assert(numel(unique(segments)) == max(max(segments))+1);
    fp = fopen([save_path, images{id}(1:end-4), '_seg.dat'], 'wb');
    [hh, ww] = size(segments);
    fprintf(fp, '%d %d %d\n', hh, ww, numel(unique(segments)));
    
    for hi = 1:hh
        for wi = 1:ww
            fprintf(fp, '%d ', segments(hi,wi));
        end
        fprintf(fp, '\n');
    end
    fclose(fp);
    
    display = 0;
    if display == 1
        % overaly segmentation
        [sx,sy]=vl_grad(double(segments), 'type', 'forward') ;
        s = find(sx | sy) ;
        imp = im ;
        imp([s s+numel(im(:,:,1)) s+2*numel(im(:,:,1))]) = 0 ;
        imagesc(imp) ; axis image off ;
        pause;
    end
    
%     % features for superpixels
%     num = max(max(segments))+1;
%     feats = zeros(num, fdim);
%     [px,py,~] = size(pixelfeats);
%     for sp = 1:num
%         pixelfeats = reshape(pixelfeats, [px*py, fdim]);
%         findex = segments == (sp-1);
%         findex = findex(:);
%         feats(sp,:) = sum(pixelfeats(findex,:),1) / sum(findex);
%     end
%     save([save_path, images{id}, '_feat.mat'], 'feats');
end
