function im = grid_to_img(post)

[h,~] = size(post);
h = sqrt(h);
w = h;

im = zeros(h,w,3);

r = post(:,1);
g = post(:,2);
b = post(:,3);

im(:,:,1) = reshape(r,h,w);
im(:,:,2) = reshape(g,h,w);
im(:,:,3) = reshape(b,h,w);