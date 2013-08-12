function visualizeParseTree3D(imgData,Wbot,W,Wout,Wcat,params,hLimit,subtree)
% hLimit: number of non-terminal nodes to be visualized
% subtree: the root node of the subtree, then only visualize the subtree
% example: visualizeParseTree3D(allData{i},Wbot,W,Wout,Wcat,params,20);
% load ../output/penn-fudan-all_fullParams_hid50_PTC0.0001_fullC0.0001_L0.05.mat
% load ../data/penn-fudan-allData-train.mat

close all

if nargin < 7
    hLimit = -1;
end
if nargin < 8
    subtree = -1;
end

topCorr=0;
imgTreeTop = parseImage(topCorr,Wbot,W,Wout,Wcat,imgData.adj, ...
    imgData.feat2,imgData.segLabels,params);

numLeafNodes = size(imgData.adj,1);
numTotalNodes = size(imgTreeTop.kids,1);
nodeHeight = size(imgData.adj,1);

numLeafsUnder = ones(numLeafNodes,1);
leafsUnder = cell(numLeafNodes,1);
for s = 1:numLeafNodes
    leafsUnder{s} = s;
    nodeHeight(s) = 1;
end
    
for n = numLeafNodes+1:numTotalNodes
    kids = imgTreeTop.getKids(n);
    numLeafsUnder(n) = numLeafsUnder(kids(1))+numLeafsUnder(kids(2));
    leafsUnder{n} = [leafsUnder{kids(1)} leafsUnder{kids(2)}];
    nodeHeight(n) = nodeHeight(kids(1)) + nodeHeight(kids(2));
end

nodeHeight = nodeHeight - 1;
segs = imgData.segs2;
orderedKids = imgTreeTop.kids;

% generate centroids by using regionprop()
lCent = getCentroidSuperpixels(segs);
pCent = zeros(numTotalNodes-numLeafNodes, 2);
pn = 1;
for node = numLeafNodes+1:numTotalNodes
    leafIndex = leafsUnder{node};
    tsegs = zeros(size(segs));
    for li = 1:numel(leafIndex)
        tsegs = tsegs + bsxfun(@eq, segs, leafIndex(li));
    end
    pCent(pn,:) = getCentroidSuperpixels(tsegs);
    pn = pn + 1;
end
Cent = [lCent; pCent];
[xs,ys] = size(segs);
Cent(:,2) = xs - Cent(:,2); % NOTE: the new coordinate system is centered at left-bottom corner

if subtree < 0
    if hLimit < 0
        hLimit = numTotalNodes;
    end
    hLimit = hLimit + numLeafNodes + 1;
    if hLimit > numTotalNodes
        hLimit = numTotalNodes;
    end
    
    if hLimit > 100 + numLeafNodes
        zLimit = hLimit-numLeafNodes;
        uz = 1;
    else
        zLimit = 100;
        uz = zLimit / (hLimit-numLeafNodes);
    end

    [sx,sy]=vl_grad(double(segs), 'type', 'forward') ;
    s = find(sx | sy) ;
    imp = imgData.img ;
    imp([s s+numel(imgData.img(:,:,1)) s+2*numel(imgData.img(:,:,1))]) = 0;

    figure;
    surface(zeros(xs,ys),flipdim(imp,1),...
       'FaceColor','texturemap',...
       'EdgeColor','none',...
       'CDataMapping','scaled');
    %colormap(gray)
    view(-35,17)
    xlabel('x'); ylabel('y'); zlabel('z');
    axis([0 ys 0 xs 0 zLimit])
    axis equal
    %axis image
    hold on;

    for node = numLeafNodes+1:hLimit
        kid1 = orderedKids(node,1);
        kid2 = orderedKids(node,2);
        point = [Cent(node,:), nodeHeight(node)*uz];
        point1 = [Cent(kid1,:), nodeHeight(kid1)*uz];
        point2 = [Cent(kid2,:), nodeHeight(kid2)*uz];
        line3d(point, point1, '-b', 1, 'r'); hold on;
        line3d(point, point2, '-b', 1, 'r'); hold on;
    end
    hold off;

    % DEBUG x coordinates
    % for leaf = 1:numLeafNodes
    %     point = lCent(leaf,:);
    %     point1 = lCent(leaf,:);
    %     point(2) = xs - point(2);
    %     point1(2) = xs - point1(2);
    %     point = [point, nodeHeight(leaf)];
    %     point1 = [point1, nodeHeight(leaf)];
    %     line3d(point, point1, '-b', 1, 'r'); hold on;
    % end
    % hold off;
else
    % visualize segments
    scratch = imgData.img;
    colmap = hsv(numLeafsUnder(subtree));
    for ei = 1:numel(leafsUnder{subtree})
        col = colmap(ei,:);
        node = leafsUnder{subtree}(ei);
        
        s = scratch(:,:,1);
        s(segs == node) = s(segs == node)/3 + 100*col(1);
        scratch(:,:,1) = s;
        s = scratch(:,:,2);
        s(segs == node) = s(segs == node)/3 + 100*col(2);
        scratch(:,:,2) = s;
        s = scratch(:,:,3);
        s(segs == node) = s(segs == node)/3 + 100*col(3);
        scratch(:,:,3) = s;
    end
    
    if nodeHeight(subtree) > 100
        zLimit = nodeHeight(subtree);
        uz = 1;
    else
        zLimit = 100;
        uz = zLimit / nodeHeight(subtree);
    end
    figure;
    surface(zeros(xs,ys),flipdim(scratch,1),...
       'FaceColor','texturemap',...
       'EdgeColor','none',...
       'CDataMapping','scaled');
    %colormap(gray)
    view(-35,17)
    xlabel('x'); ylabel('y'); zlabel('z');
    axis([0 ys 0 xs 0 zLimit])
    axis equal
    %axis image
    hold on;
    
    % visualize subtree
    node = subtree;
    lineKids(node, nodeHeight, Cent, orderedKids, uz);
    hold off
end

function lineKids(node, nodeHeight, Cent, orderedKids, uz)
% recursion
    if nodeHeight(node) <= 0
        return;
    end
    kid1 = orderedKids(node,1);
    kid2 = orderedKids(node,2);
    point = [Cent(node,:), nodeHeight(node)*uz];
    point1 = [Cent(kid1,:), nodeHeight(kid1)*uz];
    point2 = [Cent(kid2,:), nodeHeight(kid2)*uz];
    line3d(point, point1, '-b', 1, 'r'); hold on;
    line3d(point, point2, '-b', 1, 'r'); hold on;
    lineKids(kid1, nodeHeight, Cent, orderedKids, uz);
    lineKids(kid2, nodeHeight, Cent, orderedKids, uz);

