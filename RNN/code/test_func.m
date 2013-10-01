%

samples = 0:10:100;
Q_all = zeros(1,length(samples));
err_all = zeros(1,length(samples));
n_cuts = zeros(1,length(samples));

k = 1;
for i = samples
    i
    [Q,rts,labels] = make_change_points(allData{1}, allTrees{1}, i);
    Q_all(k) = Q(end);
    err_all(k) = sum(labels == allData{1}.segLabels) / length(labels);
    n_cuts(k) = sum(rts>0);
    k = k + 1;
end

figure;
subplot(1,3,1); plot(samples, Q_all); legend('purity');
subplot(1,3,2); plot(samples, err_all); legend('superpixel loss');
subplot(1,3,3); plot(samples, n_cuts); legend('number of cuts');