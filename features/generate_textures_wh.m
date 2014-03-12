%

%% Preliminary

%startup_directory;
startup_weizmann_horse;

addpath(genpath(segbench_dir));

% Texton filename
texton_file = 'textons_wh.mat';

% Create filterbank
fb = fbCreate(6,1,3,sqrt(2));
visualizeFB(fb);

% Number of Textons
ntex = 64;

% files
rt_path = 'data_weizmann_horse/Color/';
files = dir(rt_path);
images = cell(length(files)-2,1);
fp = fopen('data_weizmann_horse/list.txt','w');
for j = 3:length(files)
    images{j-2} = files(j).name;
    img = imread([rt_path, images{j-2}]);
    [w,h,~] = size(img);
    fprintf(fp, '%s %d %d\n', images{j-2}(1:end-4), w, h);
end
fclose(fp);

%% Load Textons, otherwise generate them

if exist(texton_file)
    load(texton_file, 'tex');
else
    fprintf('Creating Textons.\n');
    
    % Store the images
    % NOTE : You may want to select a subset of the images instead of all the images.
    im = [];
    fim = cell(size(fb));
    
    for i=1:50:numel(images)
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

for i=1:numel(images)
    fprintf('%d\n', i);
    
    %imfn = sprintf('%s/%s/%s_%04d.jpg', lfw_dir, names{i}, names{i}, nums(i));
    imfn = [rt_path images{i}];
    
    imi = rgb2gray(double(imread(imfn)) / 255);
    
    %run pre-made filters over image at different orientations and resolutions to get
    %responses over the image imi
    fbr =  fbRun(fb, imi);
    
    map = assignTextons(fbr, tex);
    
    tmfn = sprintf('%s/%s_tex.dat', tex_dir, images{i}(1:end-4));
    fidtm = fopen(tmfn, 'w');
    
    fprintf(fidtm, '%d\n', map');
    fclose(fidtm);
end

