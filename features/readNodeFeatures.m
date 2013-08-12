% Reads the features

% returns (number of nodes, adjMat, node energies, edge energies)
% arguments (name, id, feature_directory)

function [numNodes H] = readNodeFeatures(ns, features_dir)
ffn = sprintf('%s/%s_spfn1p.dat', features_dir, ns);
fidf = fopen(ffn);

numNodes = fscanf(fidf, '%d', 1);
numNodeFeatures = fscanf(fidf, '%d', 1);

Hp = fscanf(fidf, '%f', [numNodeFeatures numNodes]);
%H = Hp(2:end,:);
H = Hp';

% numEdges = fscanf(fidf, '%d', 1);
% numEdgeFeatures = fscanf(fidf, '%d', 1);
% E = sparse(numNodes, numNodes);
% S = cell(numNodes);
% 
% for i=1:numEdges
%     a = fscanf(fidf, '%d', 1)+1;
%     b = fscanf(fidf, '%d', 1)+1;
%     E(a,b) = 1;
%     Sp = fscanf(fidf, '%f', numEdgeFeatures);
%     S{a,b} = Sp(2:end);
% end

fclose(fidf);
