%%% Calculo de Histograma
clear all, close all, clc;
Im = imread('sobreexpuesta.jpg');
[H,W,nC] = size(Im); % H num lineas W num col nC num Colores
figure(1),imshow(Im); % Info contenida en la imagen es llamado pixel
%Tricotomia: RGB a nivel luz
ImR = double(Im(:,:,1));
ImG = double(Im(:,:,2));
ImB = double(Im(:,:,3));
Int = (ImR + ImG + ImB)/3; % Intensidad (GRIS)
%figure(2), imshow(uint8(Int))
% Histograma: Cuantificacion del numero de pixeles de
% cada intensidad
hist = zeros(256,1);
n = [0:255]';
for j = 1:H
    for i = 1:W
        pix = round(Int(j, i) + 1);
        hist(pix) = hist(pix) + 1;
    end
end
P = hist/(H*W);
figure(3), plot(n, hist);
%figure(4), plot(n, P);
sum(P);
F = size(256, 1);
F(1) = P(1);
% si F(x-h) <= F(x+h) es monotona creciente
for k=2:256
   F(k) = F(k-1) + P(k);
end
figure(5), plot(n,F), grid
% escalon unitario
Us = @(t) (t>=0); 
% Funcion histograma normalizado
Ps = @(x) P(min(max(round(x),1),256)).*Us(x-1); 
% Funcion densidad
Fs = @(x) F(min(max(round(x),1),256)).*Us(x-1);
% Enhancing = mejora de contraste
% Ecualizacion de histogramas
ImR1 = zeros(size(ImR));
ImG1 = zeros(size(ImG));
ImB1 = zeros(size(ImB));

for j = 1:H
    for i = 1:W
        pixOR = round(ImR(j, i) + 1);
        ImR1(j, i) = 255*Fs(pixOR);
        pixOG = round(ImG(j, i) + 1);
        ImG1(j, i) = 255*Fs(pixOG);
        pixOB = round(ImB(j, i) + 1);
        ImB1(j, i) = 255*Fs(pixOB);
    end
end

ImColEq = uint8(round(cat(3, ImR1, ImG1, ImB1)));
figure(6), imshow(ImColEq);
