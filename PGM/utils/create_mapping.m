%%% create mapping between superpixel and grids

function [R_block, R_sp] = create_mapping(sp,num_sp,newdim,olddim)

blocksize = olddim / newdim;
blockmap = zeros(olddim, olddim);

yrange = 1:blocksize:olddim;
xrange = 1:blocksize:olddim;

for i=1:numel(yrange)
    for j=1:numel(xrange)
        
        starti = yrange(i);
        endi = round(starti + blocksize - 1);
        
        startj = xrange(j);
        endj = round(startj + blocksize - 1);
        
        starti = round(starti);
        startj = round(startj);
        
        id = sub2ind([newdim newdim], i,j);
        
        blockmap(starti:endi, startj:endj) = id;
    end
end

mapping.sp    = cell(num_sp,1);
mapping.block = cell(newdim^2,1);

for spi = 1:num_sp,
    hits = sp == spi;
    h = hist(blockmap(hits), 1:newdim^2);
    blockdist = h / sum(h);
    bids = find(blockdist);
    
    mapping.sp{spi}.ids = bids;
    mapping.sp{spi}.dist = blockdist(bids);
end

for b = 1:newdim^2,
    hits = blockmap == b;
    h = hist(sp(hits), 1:num_sp);
    spdist = h / sum(h);
    spids = find(spdist);
    mapping.block{b}.ids = spids;
    mapping.block{b}.dist = spdist(spids);
end

num_sp = length(mapping.sp);
num_block = length(mapping.block);

R_block = zeros(num_block,num_sp);
for blocki = 1:num_block,
    sp_ids = mapping.block{blocki}.ids;
    dist = mapping.block{blocki}.dist;
    R_block(blocki,sp_ids) = dist;
end

R_sp = zeros(num_block,num_sp);
for spi = 1:num_sp,
    bl_ids = mapping.sp{spi}.ids;
    dist = mapping.sp{spi}.dist;
    R_sp(bl_ids,spi) = dist;
end

return;
