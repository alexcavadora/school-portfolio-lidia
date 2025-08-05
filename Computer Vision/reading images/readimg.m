close all, clear all, clc;

img = imread('seals.jpg');

[height, width, colors] = size(img)

# We have 3 channels on each pixel the width x height grid
# Each channel can store values from 0-255 (uint8)€
# These three channels represent Red, Green and Blue
# This approach for storing colors is called Trichotomy
figure(1), imshow(img);
title("Original image")

typeinfo(img)

# Extracting channels accordingly
red_channel = double(img(:, :, 1));
green_channel = double(img(:, :, 2));
blue_channel = double(img(:, :, 3));

# Creating a zeros channel with proper dimensions to zero out channels individually
zeros_channel = zeros(size(red_channel));

# Creating images with the extracted channels and zeros channel

# Red channel only image
img_red = uint8(cat(3, red_channel, zeros_channel, zeros_channel));
#figure(2), imshow(img_red), title("Red channel");

# Blue channel only image
img_blue = uint8(cat(3, zeros_channel, blue_channel, zeros_channel));
#figure(3), imshow(img_blue), title("Blue channel");

# Green channel only image
img_green = uint8(cat(3, zeros_channel, zeros_channel, green_channel));
#figure(4), imshow(img_green), title("Green channel");

# Red and green to represent yellow channel

img_yellow = uint8(cat(3, red_channel, green_channel, zeros_channel));
figure(5), imshow(img_yellow), title("Yellow channel");

# Image reconstruction using all channels again
img_reconstruct = uint8(cat(3,red_channel, green_channel, blue_channel));
#figure(6), imshow(img_reconstruct), title("Reconstructed image");

# Abstract images
img_abstract1 = uint8(cat(3,green_channel, blue_channel, red_channel));
#figure(7), imshow(img_abstract1), title("Abstract Image");

img_abstract2 = uint8(cat(3,blue_channel, red_channel, green_channel));
#figure(8), imshow(img_abstract2), title("Abstract Image");


