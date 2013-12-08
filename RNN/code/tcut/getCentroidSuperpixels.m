function centroids = getCentroidSuperpixels(segs)

s = regionprops(segs, 'Centroid');

centroids = cat(1, s.Centroid);

%imagesc(segs), hold on, plot(centroids(:,1), centroids(:,2), 'k*'), hold off

% figure;
% im = allData{1}.img;
% [sx,sy]=vl_grad(double(segs), 'type', 'forward') ;
% s = find(sx | sy) ;
% imp = im ;
% imp([s s+numel(im(:,:,1)) s+2*numel(im(:,:,1))]) = 0;
% imagesc(imp), hold on, plot(centroids(:,1), centroids(:,2), 'b*'), hold off
% axis equal