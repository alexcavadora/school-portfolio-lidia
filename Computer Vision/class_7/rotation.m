function main
    clear all, close all, clc;
    %Im = imread('jasgvdas.jpeg');
    Im = imread('peppers.png');
    [H,W,nC] = size(Im); % H num lineas W num col nC num Colores
    figure(1),imshow(Im); % Info contenida en la imagen es llamado pixel
    %Tricotomia: RGB a nivel luz
    ImR = double(Im(:,:,1));
    ImG = double(Im(:,:,2));
    ImB = double(Im(:,:,3));

    r = @(th) [cos(th) sin(th);
               -sin(th) cos(th)]

    ImRo = zeros(size(ImR));
    ImGo = zeros(size(ImG));
    ImBo = zeros(size(ImB));

    th = 90*pi/180;

    for Y0 = 1:H
      for X0 = 1:W
        P1 = r(th) *[X0-W/2; Y0-H/2];
        X1 = round(P1(1))+W/2;
        Y1 = round(P1(2))+H/2;

        if X1 <= W && X1 >= 1 && Y1 <= H && Y1 >= 1
          ImRo(Y1, X1) = ImR(Y0, X0);
          ImGo(Y1, X1) = ImG(Y0, X0);
          ImBo(Y1, X1) = ImB(Y0, X0);
        endif
      endfor
    endfor
    ImF= uint8(cat(3,ImRo,ImGo,ImBo));
    figure(1), imshow(ImF);
end

function [P, G] = histG(Im)
    [H,W,nC] = size(Im);
    hist = zeros(256,1);
    G = zeros(256,1);
    for j = 1:H
        for i = 1:W
            pix = round(Im(j, i) + 1);
            hist(pix) = hist(pix) + 1;
        end
    end
    P = hist/(H*W);

    G(1) = P(1);
    for k=2:256
       G(k) = G(k-1) + P(k);
    end
end
