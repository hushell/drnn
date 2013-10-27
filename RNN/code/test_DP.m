function [Q_all, err_all, n_cuts] = test_DP(imgData, tree, n_labs, samples, vis)
% testing DP with different lambda

if nargin < 4
    samples = 0:10:100;
    vis = 0;
end

if nargin < 5
    vis = 0;
end

Q_all = zeros(1,length(samples));
err_all = zeros(1,length(samples));
n_cuts = zeros(1,length(samples));

k = 1;
for i = samples
    tic;
    %[Q,cuts,labels] = make_change_points(imgData, tree, n_labs, i);
    %[Q,cuts,labels] = merge_cut(imgData, tree, n_labs, i);
    %[Q,cuts,labels] = majority_passing(imgData, tree, n_labs, i);
    [Q,cuts,labels] = mc_propagation(imgData, tree, n_labs, i);
    fprintf('lambda = %d, took %f seconds.\n', i, toc);
    Q_all(k) = Q(end);
    err_all(k) = sum(labels == imgData.segLabels) / length(labels);
    n_cuts(k) = sum(cuts>0);
    k = k + 1;
end

n_pixels = size(imgData.img,1)*size(imgData.img,2);

if vis
    %figure;
    subplot(1,3,1); plot(samples, Q_all ./ n_pixels, '-go', 'linewidth', 2); legend('pixel level loss');
    subplot(1,3,2); plot(samples, err_all, '-bo', 'linewidth', 2); legend('superpixel level loss');
    subplot(1,3,3); plot(samples, n_cuts, '-ro', 'linewidth', 2); legend('number of cuts');
end