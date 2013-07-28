%

%% Preliminary

%startup_directory;
startup_penn_fudan;

addpath(genpath(segbench_dir));

% Texton filename
texton_file = 'textons_pf.mat';

% Create filterbank
fb = fbCreate(6,1,3,sqrt(2));

% Number of Textons
ntex = 64;

%% Load Textons, otherwise generate them

if exist(texton_file)
    load(texton_file, 'tex');
else

    fprintf('Creating Textons.\n');
    
    rt_path = '/home/hushell/working/deep/datasets/pedestrian_parsing/Color/';
    files = dir(rt_path);
    images = cell(length(files)-2,1);
    for j = 3:length(files)
        images{j-2} = files(j).name;
    end
    
    % Store the images
    % NOTE : You may want to select a subset of the images instead of all the images.
    im = [];
    fim = cell(size(fb));
    
    for i=1:numel(images)
        imfn = rgb2gray(double(imread([rt_path, images{i}])) / 255);
        fimfn = fbRun(fb, imfn);
        for fi = 1:numel(fim)
            fim{fi} = [fim{fi}, fimfn{fi}(:)'];
        end
        %im = [im rgb2gray(double(imread(imfn)) / 255)];
    end    
    
    [tmap,tex] = computeTextons(fim, ntex);
    %[tmap,tex] = computeTextons(fbRun(fb, im),ntex);
    
    save(texton_file, 'tex');
end

%% Generate texture features

fprintf('Generating Texture Features\n');

%create texture directory if it doesn't exist
if (~exist(tex_dir, 'dir'))
    mkdir(tex_dir);
end

fid = fopen(all_list);
names = {};
nums = [];

%read names and ids
while(true)
    s = fscanf(fid, '%s', 1);
    if(isempty(s))
        break
    end
    n = fscanf(fid, '%d', 1);
    names{end+1} = s;
    nums(end+1) = n;
end
fclose(fid);

for i=1:numel(nums)
    fprintf('%d\n', i);
    
    imfn = sprintf('%s/%s/%s_%04d.jpg', lfw_dir, names{i}, names{i}, nums(i));
    
    imi = rgb2gray(double(imread(imfn)) / 255);
    
    %run pre-made filters over image at different orientations and resolutions to get
    %responses over the image imi
    fbr =  fbRun(fb, imi);
    
    map = assignTextons(fbr, tex);
    
    %create directory if it doesn't exist
    imdir = [tex_dir names{i}];
    
    if (~exist(imdir, 'dir'))
        mkdir(imdir);
    end
    
    tmfn = sprintf('%s/%s/%s_%04d.dat', tex_dir, names{i}, names{i}, nums(i));
    fidtm = fopen(tmfn, 'w');
    
    fprintf(fidtm, '%d\n', map');
    fclose(fidtm);
end

