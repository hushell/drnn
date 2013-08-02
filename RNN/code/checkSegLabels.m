function checkSegLabels(segments, labels, segLabels)

figure(100),imagesc(labels)
for j = 1:max(segLabels)    
    A = find(segLabels == j);
    L = zeros(size(segments));
    for i = 1:numel(A)
    L = L + (segments == A(i));
    end
    figure, imagesc(L ~= 0)
end