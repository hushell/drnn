function visualizeOneTreeImg(imgData,Wbot,W,Wout,Wcat,params,visuFolder,num,segCatsOrTree)
% segCatsOrTree==1: prints the segment labels
% segCatsOrTree==2: prints the 2..n merged segments
topCorr=0;
imgTreeTop = parseImage(topCorr,Wbot,W,Wout,Wcat,imgData.adj, ...
    imgData.feat2,imgData.segLabels,params);

numLeafNodes = size(imgData.adj,1);
numTotalNodes = size(imgTreeTop.kids,1);
% TODO: fix that imgTreeTop.nodeFeatures has 1 too many features!

if segCatsOrTree==1
    %1 sky         0,
    %2 tree        2,
    %3 road        1,
    %4 grass       1,
    %5 water       1,
    %6 building    2,
    %7 mountain    2,
    %8 foreground  2
    colmap = [...
    0.8000    0.8000    0.8000;... % 1 grey
    0.4196    0.5569    0.1373;... % 2 dark green
    0.5451    0.1333    0.3216;... % 3 VioletRed4
    0         1.0000         0;... % 4 normal green
         0         0    1.0000;... % 5 blue
    1.0000         0         0;... % 6 red
    0.5451    0.2706    0.0745;... % 7 SaddleBrown
    1.0000    0.6471         0;... % 8 Orange
    ];
    
% saveTo = [visuFolder 'imgLabels' num2str(num)];
% colorImgWithLabels(imgData.segs2,imgData.img,imgTreeTop.nodeCat,colmap,saveTo);
colorImgWithLabels_vlfeat(imgData.segs2,imgData.labels,imgTreeTop.nodeCat,imgData.segLabels, imgData.img);
    
    
elseif segCatsOrTree==2
    
    numLeafsUnder = ones(numLeafNodes,1);
    leafsUnder = cell(numLeafNodes,1);
    for s = 1:numLeafNodes
        leafsUnder{s} = s;
    end
    
    for n = numLeafNodes+1:numTotalNodes
        kids = imgTreeTop.getKids(n);
        numLeafsUnder(n) = numLeafsUnder(kids(1))+numLeafsUnder(kids(2));
        leafsUnder{n} = [leafsUnder{kids(1)} leafsUnder{kids(2)}];
    end
    
    %clear visuFolder;
    
    for hasLeafs = 2:numTotalNodes
        segList = find(numLeafsUnder==hasLeafs);
        if ~isempty(segList)
            %if exist('visuFolder','var')
            if ~isempty(visuFolder)
                saveTo = [visuFolder 'img' num2str(num) '_superSegs' num2str(hasLeafs)];
                %[~,~] = visualizeSegments(imgData.segs2,imgData.img, leafsUnder(segList),imgTreeTop.nodeLabels(:,segList),saveTo);
                [~,~] = visualizeSegments(imgData.segs2,imgData.img, leafsUnder(segList),[],saveTo);
            else
                [~,~] = visualizeSegments(imgData.segs2,imgData.img, leafsUnder(segList),[]);
            end
        end
        %pause(1);
        pause;
    end
    
elseif segCatsOrTree==3
    %1 hair
    %2 face
    %3 ucloth 
    %4 lcloth 
    %5 arms  
    %6 legs
    %7 BG 
    colmap = [...
    0.5451    0.2706    0.0745;... % 1 SaddleBrown
    1.0000    0.6471         0;... % 2 Orange
    0         1.0000         0;... % 3 normal green
    0.5451    0.1333    0.3216;... % 4 VioletRed4    
    0.4196    0.5569    0.1373;... % 5 dark green
         0         0    1.0000;... % 6 blue
    1.0000         0         0;... % 7 red
    ];
    
saveTo = [visuFolder 'imgLabels' num2str(num)];
colorImgWithLabels(imgData.segs2,imgData.img,imgTreeTop.nodeCat,colmap,saveTo);
            
end
