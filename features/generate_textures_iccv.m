%

%% Preliminary

%startup_directory;
startup_iccv09;

addpath(genpath(segbench_dir));

% Texton filename
texton_file = 'textons_iccv.mat';

% Create filterbank
fb = fbCreate(6,1,3,sqrt(2));
visualizeFB(fb);

% Number of Textons
ntex = 64;

%% Load Textons, otherwise generate them

if exist(texton_file)
    load(texton_file, 'tex');
else
    fprintf('Creating Textons.\n');
    
    % Store the images
    % NOTE : You may want to select a subset of the images instead of all the images.
    im = [];
    fim = cell(size(fb));
    
    step = 100;
    maxsample = 200*200;
    for i=1:5:length(allData)
      fprintf('%d\n', i);
        imfn = rgb2gray(double(allData{i}.img) / 255);
        fimfn = fbRun(fb, imfn);
        for fi = 1:numel(fim)
            fim{fi} = [fim{fi}, fimfn{fi}(1:step:maxsample)'];
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

for i=1:length(allData)
    fprintf('%d\n', i);
    
    %imfn = sprintf('%s/%s/%s_%04d.jpg', lfw_dir, names{i}, names{i}, nums(i));
    %imfn = [rt_path images{i}];
    
    imi = rgb2gray(double(allData{i}.img) / 255);
    
    %run pre-made filters over image at different orientations and resolutions to get
    %responses over the image imi
    fbr =  fbRun(fb, imi);
    
    map = assignTextons(fbr, tex);
    
    tmfn = sprintf('%s/%s_tex.dat', tex_dir, ['iccv09_' num2str(i)]);
    fidtm = fopen(tmfn, 'w');
    
    fprintf(fidtm, '%d\n', map');
    fclose(fidtm);
end

