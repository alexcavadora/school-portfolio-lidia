from cv2 import imread as cvimread, imshow as cvimshow, cvtColor, waitKey, destroyAllWindows, COLOR_BGR2RGB, COLOR_RGB2BGR
from numpy import zeros as zrs, uint8, stack, zeros_like, multiply
import matplotlib.pyplot as plt
from math import floor
imread = lambda Im: cvtColor(cvimread(Im), COLOR_BGR2RGB)
size = lambda im: im.shape
imshow = lambda title,im: cvimshow(title,cvtColor(im, COLOR_RGB2BGR))
double = lambda im: im.astype(float)
cat = lambda imR, imG, imB: stack([imR, imG, imB], axis=2)
plot = lambda title, n, hist: (plt.figure(title),plt.plot(n, hist),plt.grid(), plt.show())
zeros = lambda a, b : zrs((a,b))

# Read the image
Im = imread('peppers.png')

H, W, nC = size(Im)
#print(f"Height: {H}, Width: {W}, Channels: {nC}")

imshow('1',Im)
# Split RGB channels
ImR = double(Im[:, :, 0])
ImG = double(Im[:, :, 1])
ImB = double(Im[:, :, 2])


n = range(0, 256)
histR = zeros(256,1);
histG = zeros(256,1);
histB = zeros(256,1);

for j in range(H):
    for i in range(W):
        pix = uint8(ImR[j, i])
        histR[pix] = histR[pix] + 1

        pix = uint8(ImG[j, i])
        histG[pix] = histG[pix] + 1

        pix = uint8(ImB[j, i])
        histB[pix] = histB[pix] + 1

pR = histR/(H*W)
pG = histG/(H*W)
pB = histB/(H*W)

FR = zeros(256,1);
FG = zeros(256,1);
FB = zeros(256,1);

FR[0] = pR[0] 
FG[0] = pG[0] 
FB[0] = pB[0] 

for k in range(1, 256):
    FR[k] = FR[k-1] + pR[k]
    FG[k] = FG[k-1] + pG[k]
    FB[k] = FB[k-1] + pB[k]


plot('2', n, FR)
plot('3', n, FG)
plot('4', n, FB)

h, w = size(ImR)
ImR1 = zeros(h, w)
ImG1 = zeros(h, w)
ImB1 = zeros(h, w)

for j in range(H):
    for i in range(W):
        pix = round(ImR[j,i])
        ImR1[j,i] = 255.0*FR[pix, 0]
        pix = round(ImG[j,i])
        ImG1[j,i] = 255.0*FG[pix, 0]
        pix = round(ImB[j,i])
        ImB1[j,i] = 255.0*FB[pix, 0]

ImColeq = uint8(cat(ImR1,ImG1,ImB1))
imshow('5', ImColeq)

waitKey(0)
destroyAllWindows()