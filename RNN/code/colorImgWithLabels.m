function [numLabels scratch] = colorImgWithLabels(segMap, labelim, nodeLabels, im, segments)

[h w] = size(segMap);

%fullMask = zeros(h,w);
%wordsForMask = {'bla'};

scratch = labelim;
numLabels = 0;
colorCat = zeros(h,w);

%numSegs = max(segMap(:));
for i = 1:length(nodeLabels)
    colorCat(segMap == i) = nodeLabels(i);
end

[sx,sy]=vl_grad(double(segments), 'type', 'forward') ;
s = find(sx | sy) ;
imp = im ;
imp([s s+numel(im(:,:,1)) s+2*numel(im(:,:,1))]) = 0 ;
subplot(1,4,1); imagesc(imp) ; axis image off ;
subplot(1,4,2); imshow(label2rgb(colorCat));
subplot(1,4,3); imshow(label2rgb(labelim));
subplot(1,4,4); imshow(im);
pause
%imshow(label2rgb(colorCat));
%imshow(scratch);


end