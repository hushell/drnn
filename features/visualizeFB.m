function visualizeFB(fb)

[h w] = size(fb);

ind = 1;
for hi = 1:h
    for wi = 1:w
        subplot(w,h,ind); imagesc(fb{hi,wi}); axis image off
        ind = ind + 1;
    end
end
colormap(gray)

