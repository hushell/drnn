function [pixel_label] = create_pixel_label_from_ppm(I)

ir = double(reshape(I, [250^2 3]));

% 4 x 3 where each row is hair/skin/clothing/bg
labelids = [0 0 0; 255 0 0; 0 255 0; 0 0 255];

pixel_label = zeros(250,250);

for i=1:size(labelids,1)
    hits = ismember(ir, labelids(i,:), 'rows');
    pixel_label(hits) = i;
end

return;