% extract features for superpixels by averaging and normalizing features of
% each pixel inside the superpixel
clear
close all
run('/home/hushell/working/deep/vlfeat-0.9.16/toolbox/vl_setup');

load ../RNN/data/iccv09-allData-train.mat

save_path = 'data_iccv09/feat_superpixels/';

for id = 1:length(allData)
    
    %segments = vl_slic(im, regionSize, regularizer) ;
    %segments = allData{id}.segs2;
    file = ['../BSR/iccv09data_train/ucm2/iccv09_train_' num2str(id) '.mat'];
    load(file);
    ucm = ucm2(3:2:end, 3:2:end);
    regions = bwlabel(ucm == 0,4);
    regions = padarray(regions,[1 1]);
    el = strel('diamond',1); 
    for i = 1:2
       tmp = imdilate(regions,el);
       regions(regions == 0) = tmp(regions == 0);
    end
    segments = regions(2:end-1,2:end-1);
    
    %assert(numel(unique(segments)) == max(max(segments))+1);
    fp = fopen([save_path, 'iccv09_', num2str(id), '_seg.dat'], 'wb');
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
        im = allData{id}.img ;
        imp = im;
        imp([s s+numel(im(:,:,1)) s+2*numel(im(:,:,1))]) = 0 ;
        imagesc(imp) ; axis image off ;
        pause;
    end
end
