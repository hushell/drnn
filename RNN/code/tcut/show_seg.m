function show_seg(imgData, imgTreeTop, theta_plus, n_labs, p)
% useless function


[Q,cuts,labels] = tree_cut(imgData, imgTreeTop, theta_plus, n_labs, p, 1);

% figure;
% colorImgWithLabels_vlfeat(imgData.segs2, imgData.labels, labels, imgData.segLabels, imgData.img);
% 
% [w,h,~] = size(imgData.img);
% seg_im = zeros(w,h);
% for i = 1:length(labels)
%     seg_im(imgData.segs2 == i) = labels(i);
% end
% 
% figure;
% imagesc(seg_im);

