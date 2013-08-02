%%% --- addpath --- %%%
addpath data/;
addpath minFunc/;
addpath utils/;
addpath common/;

%%% --- directory to store model parameters --- %%%
fsave_dir = 'weights';
if ~exist(fsave_dir,'dir'),
    mkdir(fsave_dir);
end

%%% --- directory for log --- %%%
log_dir = 'log';
if ~exist(log_dir,'dir'),
    mkdir(log_dir);
end

%%% --- data directories --- %%%
startup_directory;


%%%  --- training, validation, and test files --- %%%
train_file = 'parts_train.txt';
valid_file = 'parts_validation.txt';
test_file = 'parts_test.txt';

% training
fidtrain = fopen(train_file);
trainnames = {};
trainnums = [];
while(true)
    s = fscanf(fidtrain, '%s', 1);
    if(isempty(s))
        break
    end
    n = fscanf(fidtrain, '%d', 1);
    trainnames{end+1} = s;
    trainnums(end+1) = n;
end
fclose(fidtrain);
num_train = numel(trainnums);

% validation
fidvalid = fopen(valid_file);
validnames = {};
validnums = [];
while(true)
    s = fscanf(fidvalid, '%s', 1);
    if(isempty(s))
        break
    end
    n = fscanf(fidvalid, '%d', 1);
    validnames{end+1} = s;
    validnums(end+1) = n;
end
fclose(fidvalid);
num_valid = numel(validnums);

% testing
fidtest = fopen(test_file);
testnames = {};
testnums = [];
while(true)
    s = fscanf(fidtest, '%s', 1);
    if(isempty(s))
        break
    end
    n = fscanf(fidtest, '%d', 1);
    testnames{end+1} = s;
    testnums(end+1) = n;
end
fclose(fidtest);
num_test = numel(testnums);
