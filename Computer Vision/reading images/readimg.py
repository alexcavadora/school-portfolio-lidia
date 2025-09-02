from PIL import Image
import numpy as np

image_path = "seals.jpg"
image = Image.open(image_path)

img = np.array(image)
width, height, channels = img.shape
print("width:",width, ", height:", height, ", channels:", channels, ", mode:", image.mode)

red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]
zero = np.zeros_like(red)

# image recreation
merged = np.stack([red, green, blue], axis=2)
merged_image = Image.fromarray(merged)
merged_image.show()

# red channel
red_image = np.stack([red, zero, zero], axis=2)
red_image = Image.fromarray(red_image)
#red_image.show()

# green channel
green_image = np.stack([zero, green, zero], axis=2)
green_image = Image.fromarray(green_image)
#green_image.show()

# blue channel
blue_image = np.stack([zero, zero, blue], axis=2)
blue_image = Image.fromarray(blue_image)
#blue_image.show()


# yellow channel
yellow_image = np.stack([red, green, zero], axis=2)
yellow_image = Image.fromarray(yellow_image)
#yellow_image.show()


# abstract images
abstract_image = np.stack([green, blue, red], axis=2)
abstract_image = Image.fromarray(abstract_image)
abstract_image.show()

# abstract images
abstract_image = np.stack([red*2, green*2, blue*2], axis=2)
abstract_image = Image.fromarray(abstract_image)
abstract_image.show()


