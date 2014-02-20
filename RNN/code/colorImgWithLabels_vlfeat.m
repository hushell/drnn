function [numLabels scratch] = colorImgWithLabels_vlfeat(segMap, labelim, nodeLabels, gtLabels, im)

[h w] = size(segMap);

%fullMask = zeros(h,w);
%wordsForMask = {'bla'};

scratch = labelim;
numLabels = 0;
colorCat = zeros(h,w);
spColorGT = zeros(h,w);

%numSegs = max(segMap(:));
for i = 1:length(gtLabels)
    colorCat(segMap == i) = nodeLabels(i);
    spColorGT(segMap == i) = gtLabels(i);
end

[sx,sy]=vl_grad(double(segMap), 'type', 'forward') ;
s = find(sx | sy) ;
imp = im ;
imp([s s+numel(im(:,:,1)) s+2*numel(im(:,:,1))]) = 0 ;
subplot(2,2,1); imagesc(imp) ; axis image off ; title('superpixel');
subplot(2,2,2); imshow(label2rgb(colorCat)); title('prediction');
subplot(2,2,3); imshow(label2rgb(spColorGT)); title('sp GT');
subplot(2,2,4); imshow(label2rgb(labelim)); title('GT');

%pause
%imshow(label2rgb(colorCat));
%imshow(scratch);


end
