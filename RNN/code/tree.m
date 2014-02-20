classdef tree
    
    properties
        % parent pointers
        pp = [];
        nodeNames;
        nodeFeatures;
        leafFeatures=[];
        % the parent pointers do not save which is the left and right child of each node, hence:
        % numNodes x 2 matrix of kids, [0 0] for leaf nodes
        kids = [];
        % matrix (maybe sparse) with L x S, L = number of unique labels, S= number of segments
        % ground truth:
        nodeLabels=[];
        % categories: computed activations (not softmaxed)
        catAct = [];
        catOut = [];
        % computed category
        nodeCat = [];
        
        % if we have the ground truth, this vector tells us which leaf labels were correctly classified
        nodeCatsRight=0;
        
        
        % for structure prediction we want to maximize the scores
        score=0;
        % for optimizing the labeled structure, we minimize the cost
        cost=0;
        nodeScores=[];

    end
    
    
    methods
        function id = getTopNode(obj)
            id = find(obj.pp==0);
        end
        
        function kids = getKids(obj,node)
            %kids = find(obj.pp==node);
            kids = obj.kids(node,:);
        end

        function p = getParent(obj,node)
            %kids = find(obj.pp==node);
            if node>0
                p = obj.pp(node);
            else
                p=-1;
            end
        end        
        
        %TODO: maybe compute leaf-node-ness once and then just check for it
        function l = isLeaf(obj,node)
            l = ~any(obj.pp==node);
        end        
        
        
        
        function plotTree(obj,postpone)
            %TREEPLOT Plot picture of tree.
            %   TREEPLOT(p) plots a picture of a tree given a row vector of
            %   parent pointers, with p(i) == 0 for a root and labels on each node.
            %
            %   Example:
            %      myTreeplot([2 4 2 0 6 4 6],{'i' 'like' 'labels' 'on' 'pretty' 'trees' '.'})
            %   returns a binary tree with labels.
            %
            %   Copyright 1984-2004 The MathWorks, Inc.
            %   $Revision: 5.12.4.2 $  $Date: 2004/06/25 18:52:28 $
            %   Modified by Richard @ Socher . org to display text labels
            %   Modified by Shell Hu
            %   TODO: 1) visualize forest; 2) vis cuts; to check if
            %   multiple optimal solutions have the same number of cuts
                
            if nargin < 2
                postpone = zeros(length(obj.nodeNames));
            end
            
            p = obj.pp';
            [x,y,h]=treelayout(p);
            f = find(p~=0);
            ppf = p(f);
            X = [x(f); x(ppf); NaN(size(f))];
            Y = [y(f); y(ppf); NaN(size(f))];
            X = X(:);
            Y = Y(:);
            
            n = length(p);
            if n < 500,
                plot (x, y, 'wo', X, Y, 'b-');
            else
                plot (X, Y, 'r-');
            end;
            xlabel(['height = ' int2str(h)]);
            axis([0 1 0 1]);
            
            if ~isempty(obj.nodeNames)
                for l=1:length(obj.nodeNames)

            		if postpone(l)
            		    pcolor = [1 1 0];
            		else
            		    pcolor = [1 1 .6];
            		end

                    if isnumeric(obj.nodeNames(l))
                        text(x(l),y(l),num2str(obj.nodeNames(l)),'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                    else
                        text(x(l),y(l),obj.nodeNames{l},'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                    end
                    if ~isempty(obj.nodeLabels)
                        if iscell(obj.nodeNames)
                            text(x(l),y(l),[labels{l} '(' obj.nodeLabels{l} ')'],'Interpreter','none',...
                                'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                        else
                            % for numbers
                            if isnumeric(obj.nodeLabels(l))
%                                if isinteger(obj.nodeLabels(l))
                                     allL = obj.nodeLabels(:,l);
                                     allL = find(allL);
                                     if isempty(allL)
                                         text(x(l),y(l),[num2str(obj.nodeNames(l))],'Interpreter','none',...
                                         'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                                     else
                                         text(x(l),y(l),[num2str(obj.nodeNames(l)) ' (' mat2str(allL) ')'],'Interpreter','none',...
                                             'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                                     end
                                   
%                                else
%                                    text(x(l),y(l),[obj.nodeLabels(l) ' ' num2str(obj.nodeLabels(l),'%.1f') ],'Interpreter','none',...
%                                        'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
%                                end
                                 % change to font size 6 for nicer tree prints
                            else
                                text(x(l),y(l),[obj.nodeNames{l}],'Interpreter','none',...
                                    'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                            end
                        end
                    end
                end
            end

        end % plotTree
        
        
        function plotForest(obj,forest)
            %TREEPLOT Plot picture of tree.
            %   TREEPLOT(p) plots a picture of a tree given a row vector of
            %   parent pointers, with p(i) == 0 for a root and labels on each node.
            %
            %   Example:
            %      myTreeplot([2 4 2 0 6 4 6],{'i' 'like' 'labels' 'on' 'pretty' 'trees' '.'})
            %   returns a binary tree with labels.
            %
            %   Copyright 1984-2004 The MathWorks, Inc.
            %   $Revision: 5.12.4.2 $  $Date: 2004/06/25 18:52:28 $
            %   Modified by Richard @ Socher . org to display text labels
            %   Modified by Shell Hu
            %   TODO: 1) visualize forest; 2) vis cuts; to check if
            %   multiple optimal solutions have the same number of cuts
                
            if nargin < 2
                forest = zeros(length(obj.nodeNames));
            end
            
            p = obj.pp';
            [x,y,h]=treelayout(p);
            f = find(p~=0);
            ppf = p(f);
            X = [x(f); x(ppf); NaN(size(f))];
            Y = [y(f); y(ppf); NaN(size(f))];
            X = X(:);
            Y = Y(:);
            
            n = length(p);
            if n < 500,
                plot (x, y, 'wo', X, Y, 'b-');
            else
                plot (X, Y, 'r-');
            end;
            xlabel(['height = ' int2str(h)]);
            axis([0 1 0 1]);
            
            if ~isempty(obj.nodeNames)
                
                lforest = unique(forest);
                colmap = zeros(length(obj.nodeNames),3);
                tcolmap = hsv(numel(lforest));
                for j=1:numel(lforest)
                    tind = find(forest == lforest(j));
                    colmap(tind,:) = repmat(tcolmap(j,:),length(tind),1);
                end
                
                for l=1:length(obj.nodeNames)

                    pcolor = colmap(forest(l),:);

                    if isnumeric(obj.nodeNames(l))
                        text(x(l),y(l),num2str(obj.nodeNames(l)),'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                    else
                        text(x(l),y(l),obj.nodeNames{l},'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                    end
                    if ~isempty(obj.nodeLabels)
                        if iscell(obj.nodeNames)
                            text(x(l),y(l),[labels{l} '(' obj.nodeLabels{l} ')'],'Interpreter','none',...
                                'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                        else
                            % for numbers
                            if isnumeric(obj.nodeLabels(l))
%                                if isinteger(obj.nodeLabels(l))
                                     allL = obj.nodeLabels(:,l);
                                     allL = find(allL);
                                     if isempty(allL)
                                         text(x(l),y(l),[num2str(obj.nodeNames(l))],'Interpreter','none',...
                                         'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                                     else
                                         text(x(l),y(l),[num2str(obj.nodeNames(l)) ' (' mat2str(allL) ')'],'Interpreter','none',...
                                             'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                                     end
                                   
%                                else
%                                    text(x(l),y(l),[obj.nodeLabels(l) ' ' num2str(obj.nodeLabels(l),'%.1f') ],'Interpreter','none',...
%                                        'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
%                                end
                                 % change to font size 6 for nicer tree prints
                            else
                                text(x(l),y(l),[obj.nodeNames{l}],'Interpreter','none',...
                                    'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                            end
                        end
                    end
                end
            end

        end % plotForest
        
        
        function plotLabs(obj,labs)
            %TREEPLOT Plot picture of tree.
            %   TREEPLOT(p) plots a picture of a tree given a row vector of
            %   parent pointers, with p(i) == 0 for a root and labels on each node.
            %
            %   Example:
            %      myTreeplot([2 4 2 0 6 4 6],{'i' 'like' 'labels' 'on' 'pretty' 'trees' '.'})
            %   returns a binary tree with labels.
            %
            %   Copyright 1984-2004 The MathWorks, Inc.
            %   $Revision: 5.12.4.2 $  $Date: 2004/06/25 18:52:28 $
            %   Modified by Richard @ Socher . org to display text labels
            %   Modified by Shell Hu
            %   TODO: 1) visualize forest; 2) vis cuts; to check if
            %   multiple optimal solutions have the same number of cuts
                
            if nargin < 2
                labs = zeros(length(obj.nodeNames));
            end
            
            p = obj.pp';
            [x,y,h]=treelayout(p);
            f = find(p~=0);
            ppf = p(f);
            X = [x(f); x(ppf); NaN(size(f))];
            Y = [y(f); y(ppf); NaN(size(f))];
            X = X(:);
            Y = Y(:);
            
            n = length(p);
            if n < 500,
                plot (x, y, 'wo', X, Y, 'b-');
            else
                plot (X, Y, 'r-');
            end;
            xlabel(['height = ' int2str(h)]);
            axis([0 1 0 1]);
            
            if ~isempty(obj.nodeNames)
                
                llabs = unique(labs);
                colmap = zeros(length(obj.nodeNames),3);
                tcolmap = hsv(numel(llabs));
                for j=1:numel(llabs)
                    tind = find(labs == llabs(j));
                    colmap(tind,:) = repmat(tcolmap(j,:),length(tind),1);
                end
                
                for l=1:length(obj.nodeNames)

                    pcolor = colmap(l,:);

                    if isnumeric(obj.nodeNames(l))
                        text(x(l),y(l),num2str(obj.nodeNames(l)),'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                    else
                        text(x(l),y(l),obj.nodeNames{l},'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                    end
                    if ~isempty(obj.nodeLabels)
                        if iscell(obj.nodeNames)
                            text(x(l),y(l),[labels{l} '(' obj.nodeLabels{l} ')'],'Interpreter','none',...
                                'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                        else
                            % for numbers
                            if isnumeric(obj.nodeLabels(l))
%                                if isinteger(obj.nodeLabels(l))
                                     allL = obj.nodeLabels(:,l);
                                     allL = find(allL);
                                     if isempty(allL)
                                         text(x(l),y(l),[num2str(obj.nodeNames(l))],'Interpreter','none',...
                                         'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                                     else
                                         text(x(l),y(l),[num2str(obj.nodeNames(l)) ' (' mat2str(allL) ')'],'Interpreter','none',...
                                             'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                                     end
                                   
%                                else
%                                    text(x(l),y(l),[obj.nodeLabels(l) ' ' num2str(obj.nodeLabels(l),'%.1f') ],'Interpreter','none',...
%                                        'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
%                                end
                                 % change to font size 6 for nicer tree prints
                            else
                                text(x(l),y(l),[obj.nodeNames{l}],'Interpreter','none',...
                                    'HorizontalAlignment','center','FontSize',8,'BackgroundColor',pcolor)
                            end
                        end
                    end
                end
            end

        end % plotLabs
        
    end % methods
end % class
