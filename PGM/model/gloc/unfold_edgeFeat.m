function feat = unfold_edgeFeat(feat, Dim)

if ~exist('Dim','var'), 
    Dim = 4;
end

if ~isfield(feat,'edgeFeaturesrs'),
    [xi, xj] = find(feat.adjmat > 0);
    edgeFeaturesrs = zeros(Dim,1,1,length(xi));
    for fi = 1:length(xi),
        edgeFeaturesrs(:,:,:,fi) = feat.edgeFeatures{xi(fi), xj(fi)};
    end
    feat.edgeFeaturesrs = edgeFeaturesrs;
end

[xi, xj] = find(feat.adjmat > 0);
Aii = cell(feat.numNodes,1);
Ajj = cell(feat.numNodes,1);
Ailen = cell(feat.numNodes,1);
Ajlen = cell(feat.numNodes,1);
Aji = cell(feat.numNodes,1);
Aij = cell(feat.numNodes,1);
for ei = 1:feat.numNodes,
    Aii{ei} = find(xi == ei);
    Ajj{ei} = find(xj == ei);
    Ailen{ei} = length(Aii{ei});
    Ajlen{ei} = length(Ajj{ei});
    Aji{ei} = xj(Aii{ei});
    Aij{ei} = xi(Ajj{ei});
end
Aimat = feat.edgeFeaturesrs(:,:,:,Aii{1});
Aiidx = cell(feat.numNodes,1);
Aiidx{1} = 1:Ailen{1};
idx = Ailen{1};
for ei = 2:feat.numNodes,
    Aimat(:,:,:,end+1:end+Ailen{ei}) = feat.edgeFeaturesrs(:,:,:,Aii{ei});
    Aiidx{ei} = idx+1:idx+Ailen{ei};
    idx = idx + Ailen{ei};
end
Ajmat = feat.edgeFeaturesrs(:,:,:,Ajj{1});
Ajidx = cell(feat.numNodes,1);
Ajidx{1} = 1:Ajlen{1};
idx = Ajlen{1};
for ei = 2:feat.numNodes,
    Ajmat(:,:,:,end+1:end+Ajlen{ei}) = feat.edgeFeaturesrs(:,:,:,Ajj{ei});
    Ajidx{ei} = idx+1:idx+Ajlen{ei};
    idx = idx + Ajlen{ei};
end
feat.Aij = Aij;
feat.Aji = Aji;
feat.Aiidx = Aiidx;
feat.Ajidx = Ajidx;
feat.Ailen = Ailen;
feat.Ajlen = Ajlen;
feat.Aimat = single(Aimat);
feat.Ajmat = single(Ajmat);

return;