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
    n = [0:255]';
    P1 = histG(Int);
    figure(3), plot(n, P1);
    Io = FactorG(Int, gamma);
    figure(4), imshow(uint8(round(Io)));
    P2 = histG(Io);
    figure(5), plot(n, P2);

    ImRo = FactorG(ImR, gamma);
    ImGo = FactorG(ImG, gamma);
    ImBo = FactorG(ImB, gamma);

    ImC = uint8(round(cat(3, ImRo, ImGo, ImBo)));
    figure(6), imshow(ImC);

    ImC2 = ModGamma(ImR, ImG, ImB, gamma);
    figure(7), imshow(ImC2);

    ImC3 = ModGamma2(ImR, ImG, ImB, gamma);
    figure(8), imshow(ImC3);

    % Tarea: Hacer un resumen de que es el factor gamma y para
    % que se usa. Pasar este programa en Python
    % Limpiar este codigo para solo copiar y pegar
end

function P = histG(Im)
    [H,W,nC] = size(Im);
    hist = zeros(256,1);
    for j = 1:H
        for i = 1:W
            pix = round(Im(j, i) + 1);
            hist(pix) = hist(pix) + 1;
        end
    end
    P = hist/(H*W);
end

function Io = FactorG(Im, gamma)
    Io = 255*(Im/255).^(1/gamma);
end

function Io = minmax1(Im)
    Xmin = min(min(Im));
    Xmax = max(max(Im));
    Io = 255*((Im - Xmin)/(Xmax-Xmin));
end

function Io = ModGamma(ImR, ImG, ImB, gamma)
    Int = (ImR+ImG+ImB)/3;
    ImRo = ImR.*((ImR + eps)./(Int + eps)).^(1/gamma);
    ImGo = ImG.*((ImG + eps)./(Int + eps)).^(1/gamma);
    ImBo = ImB.*((ImB + eps)./(Int + eps)).^(1/gamma);

    ImRo = minmax1(ImRo);
    ImGo = minmax1(ImGo);
    ImBo = minmax1(ImBo);

    Io = uint8(round(cat(3, ImRo, ImGo, ImBo)));
end

function Io = ModGamma2(ImR, ImG, ImB, gamma)
    Int = (ImR+ImG+ImB)/3;
    ImRo = 255*(ImR.*(ImR + eps)./(255*Int + eps)).^(1/gamma);
    ImGo = 255*(ImG.*(ImG + eps)./(255*Int + eps)).^(1/gamma);
    ImBo = 255*(ImB.*(ImB + eps)./(255*Int + eps)).^(1/gamma);

    ImRo = minmax1(ImRo);
    ImGo = minmax1(ImGo);
    ImBo = minmax1(ImBo);

    Io = uint8(round(cat(3, ImRo, ImGo, ImBo)));
end
