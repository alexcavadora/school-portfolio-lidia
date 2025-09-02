% Segmentacion. Es la agrupacion y clasificacio de pixeles conexos que
% siguen un conujunto de reglas o atributos comunes como color, textura,
% forma o contexto.

% Metodos de clasificacion: Umbralizacion, color, textura, semantica.

% Segmentacion por umbralizacion. Se basa en histograma.


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
    Int = (ImR + ImG + ImB)/3;
    figure(2), imshow(uint8(Int))
    gamma = 2;
    k = [0:255]';
    [P1, G1] = histG(Int);
    figure(3), plot(k, P1);
    figure(4), plot(k, G1);

    Th = 0.68;
    %for k = 1:255
    %    if G1(k) >= Th, break; end
    %end
    %G1(k);

    %Io = 255 * uint8(Int>k);
    %Io = uint8(Int.*(Int>k));
    %Io = uint8(Int.*(Im>k));
    %Io = uint8(Int.*(Im>k));
    %IoR = uint8(ImR.*(ImR>k));
    %IoG = uint8(ImG.*(ImG>k));
    %IoB = uint8(ImB.*(ImB>k));

    %Io = cat(3, IoR, IoG, IoB);
    %Io = uint8(Int.*(Int>k));

    W0 = G1(Th);
    W1 = 1 - G1(Th);

    MuT = sum(k.*P1)
    VarT = sum(((k-MuT).^2).*P1)


    %figure(5), imshow(Io)
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
