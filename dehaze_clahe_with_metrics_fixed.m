%function final_output = dehaze_clahe_with_metrics_fixed(image_path)
% ===============================================================
% UNDERWATER IMAGE DEHAZING PIPELINE + METRICS (single file)
% MMLE + BASE-DETAIL + MEAN CURVATURE FILTER (Base Layer)
% + ADF (Detail Layer) + Structure-Aware Fusion
% ===============================================================
close all; clc; clear all;
addpath("Perona-Malik-Diffusion-main\");
addpath("Structure-Aware_Image_Fusion-master\");
hazy_img= im2double(imread("/Users/vijaymadaan/Downloads/paper1 2/a10.jpg"));

%% Step 2: DCP Dehazing
mmle_img = dehaze_fast(hazy_img, 0.95, 15);
figure;imshow(mmle_img);

%% Step 3: Base-Detail Layer Decomposition
detail_layer = hazy_img - mmle_img;
base_layer   = mmle_img;
figure;imshow(detail_layer);
%% Step 4: MEAN CURVATURE FILTER (Base Layer Smoothing)
iters = 3;   % recommended 20–40
dt    = 0.03; % timestep (0.1-0.2 recommended)
enhanced_baselayer = mean_curvature_filter(base_layer, iters, dt);
imwrite(enhanced_baselayer, 'BaseLayer_MCF.png');

% %% Step 5: ADF Filter (Detail Layer Enhancement)
% %% Step 5: ADF Filter (Detail Layer Enhancement) -- NO GRAYSCALE
% detail_layer_clamped = min(max(detail_layer, 0), 1);   % keep values valid
% detail_enhance = zeros(size(detail_layer_clamped));
% 
% for c = 1:3
%     detail_enhance(:,:,c) = PMD_explicit(detail_layer_clamped(:,:,c), ...
%                                          5, 1/5, 15, 1, 1);
% end


%% Step 6: Structure-Aware Fusion

f = enhanced_baselayer+detail_layer;

%% =========================

%% Display Figures
figure; imshow(hazy_img);            title('Hazy Input');
figure; imshow(mmle_img);            title('DCP (Reference)');
figure; imshow(detail_layer); title("Detail")
figure; imshow(base_layer);title("Base")
figure; imshow(enhanced_baselayer);  title('Enhanced Base (MCF)');
% figure; imshow(detail_layer);      title('Enhanced Detail (ADF)');
figure; imshow(f);        title('Fused');

%% FINAL EDGE-PRESERVING GUIDED FILTER
% =======================================================

r  = 6;        % radius (tune: 6–12)
eps = 0.02^2;  % smaller = sharper

f_final = imguidedfilter(f, f, ...
                         'NeighborhoodSize',[r r], ...
                         'DegreeOfSmoothing', eps);

figure; imshow(f_final);
imwrite(f_final,"a18f.jpg")
ref = im2double(imread("/Users/vijaymadaan/Downloads/paper1 2/ref.jpg"));
PSNR = psnr(f_final, ref);
disp(['PSNR: ', num2str(PSNR)]);
SSIM = ssim(f_final, ref);
disp(['SSIM: ', num2str(SSIM)]);

disp ("-----------------------------------------------------------");

disp("ref-final")
metricsr = main_calculate_metrics(ref, f_final);
%end


