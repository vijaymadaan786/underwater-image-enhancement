function metrics = main_calculate_metrics(original, enhanced)
    % === SIMPLE IMAGE INPUT ===
    % Just type or paste your image filenames below

    % === READ IMAGES ===
    
    % === CHECK DIMENSIONS ===
    if size(original, 3) ~= size(enhanced, 3) || any(size(original) ~= size(enhanced))
        error('Original and Enhanced images must have the same size and channels.');
    end

    % === CALCULATE METRICS ===
    metrics = calculate_all_metrics(original, enhanced);

    % === DISPLAY AND SAVE RESULTS ===
    display_metrics(metrics);
end


% === METRICS CALCULATION FUNCTION ===
function metrics = calculate_all_metrics(original, enhanced)
    metrics = struct();
    im1 = rgb2gray(original);
    im2 = rgb2gray(enhanced);

    metrics.MSE = immse(original, enhanced);
    metrics.PSNR = psnr(enhanced, original);
    metrics.SSIM = ssim(enhanced, original);
    metrics.CC = corr2(im1, im2);
    metrics.NCC = sum(sum(im1 .* im2)) / sqrt(sum(sum(im1 .^ 2)) * sum(sum(im2 .^ 2)));
    metrics.ENTROPY_enhanced = entropy(im2);
    metrics.MAE = mean2(abs(double(im1) - double(im2)));
    metrics.NAE = sum(sum(abs(double(im1) - double(im2)))) / sum(sum(double(im1)));
    metrics.STD_enhanced = std2(double(im2));
    metrics.MI = mutual_information(im1, im2);
    metrics.UIQI = uiqi(im1, im2);
    metrics.SF_enhanced = spatial_frequency(im2);
end


% === SUPPORT FUNCTIONS ===
function mi = mutual_information(img1, im2)
    img1 = round(mat2gray(img1) * 255) + 1;
    im2 = round(mat2gray(im2) * 255) + 1;
    joint_hist = accumarray([img1(:), im2(:)], 1, [256 256]);
    joint_prob = joint_hist / sum(joint_hist(:));
    prob1 = sum(joint_prob, 2);
    prob2 = sum(joint_prob, 1);
    mi = 0;
    for i = 1:256
        for j = 1:256
            if joint_prob(i, j) > 0 && prob1(i) > 0 && prob2(j) > 0
                mi = mi + joint_prob(i, j) * log2(joint_prob(i, j) / (prob1(i) * prob2(j)));
            end
        end
    end
end

function q = uiqi(img1, im2)
    img1 = double(img1);
    im2 = double(im2);
    mean1 = mean(img1(:));
    mean2 = mean(im2(:));
    std1 = std(img1(:));
    std2 = std(im2(:));
    covar = cov(img1(:), im2(:));
    covar = covar(1, 2);
    c1 = (0.01 * 255)^2;
    c2 = (0.03 * 255)^2;
    luminance = (2 * mean1 * mean2 + c1) / (mean1^2 + mean2^2 + c1);
    contrast = (2 * std1 * std2 + c2) / (std1^2 + std2^2 + c2);
    structure = (covar + c2 / 2) / (std1 * std2 + c2 / 2);
    q = luminance * contrast * structure;
end

function sf = spatial_frequency(img)
    [rows, cols] = size(img);
    rf = sqrt(sum(sum(diff(img, 1, 1).^2)) / (rows * cols));
    cf = sqrt(sum(sum(diff(img, 1, 2).^2)) / (rows * cols));
    sf = sqrt(rf^2 + cf^2);
end

function save_metrics_to_file(metrics, filename)
    fid = fopen(filename, 'w');
    fields = fieldnames(metrics);
    for i = 1:length(fields)
        fprintf(fid, '%s: %f\n', fields{i}, metrics.(fields{i}));
    end
    fclose(fid);
end

function display_metrics(metrics)
    disp(' ');
    disp('========= IMAGE QUALITY METRICS =========');
    fields = fieldnames(metrics);
    for i = 1:length(fields)
        fprintf('%20s: %.4f\n', fields{i}, metrics.(fields{i}));
    end
    disp('=========================================');
end