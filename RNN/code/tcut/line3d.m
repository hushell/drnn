function line3d(st, ed, varargin)
%

linespec = '-b';
linewid = 1;

if nargin > 2
    linespec = varargin{1};
end
if nargin > 3
    linewid = varargin{2};
end
if nargin > 4
    linespec = [linespec, 'o'];
    markedge = varargin{3};
end
if nargin > 5
    markface = varargin{4};
end
if nargin > 6
    marksize = varargin{5};
end

x1 = st(1); y1 = st(2); z1 = st(3);
x2 = ed(1); y2 = ed(2); z2 = ed(3);

a = [x1;x2];
b = [y1;y2];
c = [z1;z2];

if nargin <= 4
    plot3(a, b, c, linespec, 'LineWidth', linewid);
elseif nargin == 5
    plot3(a, b, c, linespec, 'LineWidth', linewid, ...
        'MarkerEdgeColor', markedge);
elseif nargin == 6
    plot3(a, b, c, linespec, 'LineWidth', linewid, ...
        'MarkerEdgeColor', markedge, 'MarkerFaceColor', markface);
elseif nargin >= 7
    plot3(a, b, c, linespec, 'LineWidth', linewid, ...
        'MarkerEdgeColor', markedge, 'MarkerFaceColor', markface, ...
        'MarkerSize', marksize);
end
