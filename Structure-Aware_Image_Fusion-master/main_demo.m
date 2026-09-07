clear;
close all;clc
addpath(genpath(pwd));
I1_rgb = imread('lake1.tif');
I2_rgb = imread('lake2.tif');
tic
F_rgb = zeros(size(I1_rgb), 'uint8');
for c = 1:3
    F_rgb(:,:,c) = IJF(I1_rgb(:,:,c), I2_rgb(:,:,c)); % apply fusion per channel
end

imshow(F_rgb)
%imwrite(F_rgb, 'F_echo_dtf_rgb.png'); % optional: save the RGB image


toc
matrices_new(rgb2gray(F_rgb),rgb2gray(I1_rgb),rgb2gray(I2_rgb))
xfused8 = F_rgb;
roi = drawrectangle();  % You can also use drawpolygon or drawcircle
position = round(roi.Position);  % [x, y, width, height]

% Step 3: Crop the selected region
cropped = imcrop(xfused8, position);

% Step 4: Zoom the cropped region (e.g., 2x)
zoomFactor = 2;
zoomed = imresize(cropped, zoomFactor);

% Step 5: Define where to paste the zoomed inset (e.g., top-left corner)
[xInset, yInset] = deal(20, 20);  % Adjust as needed
[hZoom, wZoom, ~] = size(zoomed);

% Step 6: Paste the zoomed region onto the original image
output = xfused8;  % Copy original
output(yInset:yInset+hZoom-1, xInset:xInset+wZoom-1, :) = zoomed;

% Step 7: Draw red rectangle on original to show selected region
output = insertShape(output, 'Rectangle', position, 'Color', 'red', 'LineWidth', 2);

% Step 8: Draw red border around inset
output = insertShape(output, 'Rectangle', [xInset, yInset, wZoom, hZoom], 'Color', 'red', 'LineWidth', 2);

% Step 9: Display final result
figure, imshow(output);