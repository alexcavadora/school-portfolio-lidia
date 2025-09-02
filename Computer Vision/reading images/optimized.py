from cv2 import imshow as cvimshow, imread as cvimread, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR, waitKey, destroyAllWindows
from numpy import shape, zeros_like as zeros, double as np_double, uint8

imread = lambda img: cvtColor(cvimread(img), COLOR_BGR2RGB)
size = lambda img: img.shape
imshow = lambda img, title: cvimshow(title, cvtColor(img, COLOR_RGB2BGR))
double = lambda img: img.as_type(np_double)


img = imread('seals.jpg')


height, width, ncolors = size(img)

print(height, width, ncolors)

imshow(img, "elpepe")

waitKey(0)
destroyAllWindows()