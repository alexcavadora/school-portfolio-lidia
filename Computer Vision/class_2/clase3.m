close all; clear all; clc;

img = imread('../reading images/seals.jpg');

[height, width, colors] = size(img);
figure(1), imshow(img);

red_channel   = double(img(:, :, 1));
green_channel = double(img(:, :, 2));
blue_channel  = double(img(:, :, 3));

intensity = (red_channel + green_channel + blue_channel) / 3;
figure(1), imshow(uint8(intensity));

histogram = zeros(256, 1);
n_column = (0:255)';

for row = 1:height
    for col = 1:width
        pixel = min(255, max(0, round(intensity(row, col)))) + 1;
        histogram(pixel) = histogram(pixel) + 1;
    end
end

figure(2), plot(n_column, histogram);

P = histogram / (height * width);
figure(3), plot(n_column, P);
