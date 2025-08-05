import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2

def closest_color_name(rgb_color):
    min_dist = float('inf')
    closest_name = None
    for name, hex_code in mcolors.CSS4_COLORS.items():
        css4_rgb = np.array(mcolors.to_rgb(hex_code))
        dist = np.linalg.norm(css4_rgb - rgb_color)
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

class FuzzyVar:
    def __init__(self, name, min_value, max_value, units, fuzzy_sets=None):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.units = units
        self.fuzzy_sets = fuzzy_sets if fuzzy_sets is not None else []

        if isinstance(self.fuzzy_sets, int):
            self.generate_fuzzy_sets(self.fuzzy_sets)

    def generate_fuzzy_sets(self, n_sets):
        self.fuzzy_sets = []
        step = (self.max_value - self.min_value) / (n_sets - 1)
        for i in range(n_sets):
                a = self.min_value + (i - 1) * step if i > 0 else self.min_value
                b = self.min_value + i * step
                c = self.min_value + (i + 1) * step if i < n_sets - 1 else self.max_value
                hue = i / (n_sets - 1)
                hsv_color = (hue, 1.0, 1.0)
                rgb_color = mcolors.hsv_to_rgb(hsv_color)
                color_name = closest_color_name(rgb_color)
                fuzzy_set = FuzzySet(f"{color_name}", a, b, c, color=rgb_color)
                self.fuzzy_sets.append(fuzzy_set)

    def membership(self, x, show=False):
        membership = []
        for fuzzy_set in self.fuzzy_sets:
            membership.append((fuzzy_set.label, fuzzy_set.membership(x)))

        if show:
            labels = [mem[0] for mem in membership if mem[1] != 0]
            values = [mem[1] for mem in membership if mem[1] != 0]
            print(f"Membresía de {x}{self.units} en {self.name}:")
            for label, value in zip(labels, values):
                print(f"\t{label}: {value:.2f}")
        return membership

    def add_fuzzy_set(self, fuzzy_set):
        self.fuzzy_sets.append(fuzzy_set)

    def remove_fuzzy_set(self, label):
        self.fuzzy_sets = [fs for fs in self.fuzzy_sets if fs.label != label]

    def modify_fuzzy_set(self, label, a=None, b=None, c=None):
        for fuzzy_set in self.fuzzy_sets:
            if fuzzy_set.label == label:
                fuzzy_set.set_parameters(a, b, c)
                break

    def plot_fuzzy_sets(self, filename='TEST.png'):
        """
        Grafica todos los conjuntos difusos de la variable.
        """
        x_values = np.linspace(self.min_value, self.max_value, 500)
        plt.figure(figsize=(8, 6))
        for fuzzy_set in self.fuzzy_sets:
            fuzzy_set.plot_membership_function(x_values)

        plt.title(f"Funciones de Membresía - {self.name}", fontsize=16)
        plt.xlabel(f"Universo de Discurso ({self.units})", fontsize=12)
        plt.ylabel("Grado de Membresía", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="upper right", fontsize=10)
        plt.xlim(self.min_value, self.max_value)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(filename)

class FuzzySet:
    def __init__(self, label, a, b, c, color):
        self.label = label
        self.a = a  # inicio de la base de la función triangular
        self.b = b  # pico de la función triangular
        self.c = c  # final de la base de la función triangular
        self.color = color  # Color de la función de membresía

    def membership(self, x):
        """
        Calcula el valor de membresía para un valor 'x'.
        """
        if x <= self.a or x >= self.c:
            return 0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b < x < self.c:
            return (self.c - x) / (self.c - self.b)

    def set_parameters(self, a, b, c):
        """
        Permite modificar los parámetros de la función de membresía triangular.
        """
        self.a = a
        self.b = b
        self.c = c

    def get_parameters(self):
        """
        Retorna los parámetros actuales de la función de membresía.
        """
        return self.a, self.b, self.c

    def plot_membership_function(self, x_values):
        y_values = [self.membership(x) for x in x_values]
        plt.plot(x_values, y_values, label=self.label, color=self.color, linewidth=2)



FuzzyHue = FuzzyVar("Hue", 0, 255, "Hue", fuzzy_sets=20)
FuzzyHue.plot_fuzzy_sets("hue.png")


image = cv2.imread('img2.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
hue_channel = hsv_image[:, :, 0]

filtered_hsv_image = np.copy(hsv_image)

filtered_hsv_image[:, :, 1] = 255
filtered_hsv_image[:, :, 2] = 255

black_threshold = 60   # Consider a pixel near black if all channels are below this threshold
white_threshold = 170   # Consider a pixel near white if all channels are above this threshold

for i in range(hue_channel.shape[0]):
    for j in range(hue_channel.shape[1]):
        hue_value = hue_channel[i, j]

        # Check for near black and near white pixels
        if np.all(image[i, j] < black_threshold):  # Near black pixel
            filtered_hsv_image[i, j] = [0, 0, 0]
        elif np.all(image[i, j] > white_threshold):  # Near white pixel
            filtered_hsv_image[i, j] = [0, 0, 255]  # Keep white as pure white in HSV
        else:
            memberships = FuzzyHue.membership(hue_value)

            # Get the fuzzy set with the maximum membership
            max_membership = max(memberships, key=lambda x: x[1])

            closest_set = [fs for fs in FuzzyHue.fuzzy_sets if fs.label == max_membership[0]][0]
            filtered_hsv_image[i, j, 0] = closest_set.b  # Apply closest fuzzy set hue

filtered_image = cv2.cvtColor(filtered_hsv_image, cv2.COLOR_HSV2BGR_FULL)
combined_image = np.hstack((image, filtered_image))
cv2.imshow('Original and Filtered Images', combined_image)
cv2.imwrite('results.png', combined_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
