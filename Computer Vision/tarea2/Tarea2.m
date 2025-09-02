clear all, close all, clc;
Im = imread('og.jpg');
[H,W,nC] = size(Im);
figure(1),imshow(Im);

ImR = double(Im(:,:,1));
ImG = double(Im(:,:,2));
ImB = double(Im(:,:,3));


histR = zeros(256,1);
histG = zeros(256,1);
histB = zeros(256,1);

n = [0:255]';
for j = 1:H
    for i = 1:W
        pix = round(ImR(j,i) + 1);
        histR(pix) = histR(pix) + 1;

        pix = round(ImG(j,i) + 1);
        histG(pix) = histG(pix) + 1;

        pix = round(ImB(j,i) + 1);
        histB(pix) = histB(pix) + 1;
    end
end


pR = histR/(H*W);
pG = histG/(H*W);
pB = histB/(H*W);


FR = zeros(256,1);
FG = zeros(256,1);
FB = zeros(256,1);


FR(1) = pR(1);
FG(1) = pG(1);
FB(1) = pB(1);


for k = 2:256
    FR(k) = FR(k-1) + pR(k);
    FG(k) = FG(k-1) + pG(k);
    FB(k) = FB(k-1) + pB(k);
end

figure(5), plot(n,FR),grid
Us = @(t) (t>=0);
%Funcion de densidad
Fs = @(x, F) F(min(max(round(x), 1),256)).*Us(x-1);



ImR1 = zeros(size(ImR));
ImG1 = zeros(size(ImG));
ImB1 = zeros(size(ImB));

for j = 1:H
    for i = 1:W
        pix = round(ImR(j,i)+1);
        ImR1(j,i) = 255.0*Fs(pix, FR);
        pix = round(ImG(j,i)+1);
        ImG1(j,i) = 255.0*Fs(pix, FG);
        pix = round(ImB(j,i)+1);
        ImB1(j,i) = 255.0*Fs(pix, FB);

    end
end
ImColeq = uint8(round(cat(3,ImR1,ImG1,ImB1)));
figure(6), imshow(ImColeq)
