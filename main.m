%% Demo of Tensor Completion
%% Demo start

% clc;
clear;close all;
rng('default');rng(1997);
addpath(genpath('utils'));
addpath(genpath('lib'));
addpath(genpath('data'));
dataName = 'face_ms';  % 'img_Einstein';
%  Please make sure the RGB image is a cubic of size [height, width, 3] and in range [0, 1].
%  You can use other tensor data such as Hyperspectral Image, Video, CT/MRI for test. 
%  Note some parameter might need reset for other methods. 
dataRoad = ['MSI/', dataName];

%% Set enable bits
Run_NTCTV            = 1;  % our method

%% Load Data 
methodName = {'Observed','NTCTV'};
Mnum = length(methodName);
load(dataRoad);  % load data
data = X;
[height, width, band] = size(data);
dim = [height, width, band];

%% Observation
i = 1;
missing_rate  = 0.95; % sampling rate, i.e, sampling_rate = 1 - missing rate

disp(['=== the missing rate is ', num2str(missing_rate), ' ===']);

sampling_rate = 1-missing_rate;
m          = round(prod(dim)*sampling_rate);
sort_dim   = randperm(prod(dim));
Omega      = sort_dim(1:m); % sampling pixels' index
Obs        = zeros(dim);
Obs(Omega) = data(Omega); % observed Img

Results{i} = Obs;
[PSNR(i), SSIM(i), FSIM(i)] = HSI_QA(255*data, 255*Results{i});

%% Run NTCTV
i = i+1;
methodName{i} = 'NTCTV';
if Run_NTCTV
    addpath(genpath(['NTCTV']));
    disp(['Running ',methodName{i}, ' ... ']);
    X0 = linear_interpolation(Obs);
    opts = [];
    for alpha = [30]
        for beta = [150]
            for r = [20]
                opts.r = r;
                opts.alpha = alpha;
                opts.beta = beta;
                tic
                [Results{i},idealY,~] = NTCTV_TC(Obs, Omega, X0, opts, data) ;
                Time(i) = toc;
                [PSNR(i), SSIM(i), FSIM(i)] = HSI_QA(255*data, 255*Results{i});
            end
        end
    end
    enList = [enList, i];
end
%% Show result
fprintf('\n');    

fprintf('================== QA Results =====================\n');
fprintf(' %8.8s    %5.5s    %5.5s    %5.5s     %5.5s  %5.5s  \n',...
    'Method', 'MPSNR', 'MSSIM', 'MFSIM',  'Time', 'Iter');

for i = 1:length(enList)
    fprintf(' %8.8s   %5.3f    %5.3f    %5.3f    %5.3f    %5.3f \n',...
        methodName{enList(i)}, PSNR(enList(i)), SSIM(enList(i)), FSIM(enList(i)), Time(enList(i)));
end