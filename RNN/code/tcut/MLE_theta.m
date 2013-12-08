function theta_plus = MLE_theta(allData, n_labs)
% MLE for theta_plus

if nargin < 2
    n_labs = 7; % for iccv09 using 8
end

counts = zeros(1,n_labs);
tcounts = zeros(1,n_labs);

for i = 1:length(allData)
    csp = allData{i}.labelCountsPerSP; % Y
    tcsp = allData{i}.numPixelInSP; % sum(Y)
    
    for k = 1:n_labs
        csp_z = csp(allData{i}.segLabels == k,:);
        tcsp_z = tcsp(allData{i}.segLabels == k);
        counts(k) = counts(k) + sum(csp_z(:,k));
        tcounts(k) = tcounts(k) + sum(tcsp_z);
    end
end

theta_plus = counts ./ tcounts;