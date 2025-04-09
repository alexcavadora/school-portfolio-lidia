import numpy as np
import random
import cv2

class AdaptiveGeometricShape:
    __slots__ = ['shape_type', 'width', 'height', 'x', 'y', 'size', 'color', 'alpha']

    def __init__(self, width, height, reference_array):
        """Initialize geometric shape with random parameters and color sampling."""
        self.width, self.height = width, height
        self.shape_type = random.randint(0, 2)
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.size = random.randint(5, 40)
        

        if random.random() < 0.7:
            try:
                sample_x = min(max(0, int(self.x)), width-1)
                sample_y = min(max(0, int(self.y)), height-1)
                self.color = tuple(reference_array[sample_y, sample_x])
            except IndexError:
                self.color = tuple(random.randint(0, 255) for _ in range(3))
        else:
            self.color = tuple(random.randint(0, 255) for _ in range(3))
        
        self.alpha = random.uniform(0.1, 0.7)

    def mutate(self, reference_array, improvement_rate=1.0):
        """Mutate shape's parameters with adaptive rate based on improvement."""
        adapt_factor = max(0.5, min(3.0, 1.0 / (improvement_rate + 0.01)))
        mutation_base_rate = 0.5
        
        mutation_params = [
            ("x", self.width, 0.2 * adapt_factor),
            ("y", self.height, 0.2 * adapt_factor),
            ("size", 100, 0.3 * adapt_factor),
            ("color", 255, 0.2 * adapt_factor),
            ("alpha", 1.0, 0.3 * adapt_factor),
            ("shape_type", 2, 0.1 * adapt_factor)
        ]
        
        for param, max_val, rate in mutation_params:
            if random.random() < mutation_base_rate * adapt_factor:
                if param == "color":
                    if random.random() < 0.3:
                        try:
                            sample_x = min(max(0, int(self.x)), self.width-1)
                            sample_y = min(max(0, int(self.y)), self.height-1)
                            self.color = tuple(reference_array[sample_y, sample_x])
                        except IndexError:
                            self.color = tuple(np.clip(c + random.randint(-60, 60), 0, 255) for c in self.color)
                    else:
                        self.color = tuple(np.clip(c + random.randint(-60, 60), 0, 255) for c in self.color)
                elif param == "shape_type":
                    self.shape_type = random.randint(0, 2)
                else:
                    current_val = getattr(self, param)
                    variation = random.gauss(0, rate * max_val)
                    setattr(self, param, np.clip(current_val + variation, 0, max_val))
        
        return self

    def draw(self, img_array):
        """Draw the shape directly on a numpy array for performance."""
        h, w, _ = img_array.shape
        

        fill_color = [int(c) for c in self.color[::-1]]
        alpha_value = int(self.alpha * 255)
        

        fill_color_with_alpha = fill_color + [alpha_value]
        

        if self.shape_type == 0:
            x0, y0 = max(0, int(self.x - self.size)), max(0, int(self.y - self.size))
            x1, y1 = min(w, int(self.x + self.size)), min(h, int(self.y + self.size))
            if x1 > x0 and y1 > y0:
                cv2.circle(img_array, (int(self.x), int(self.y)), int(self.size), 
                           fill_color_with_alpha, -1)
        

        elif self.shape_type == 1:
            x0, y0 = max(0, int(self.x)), max(0, int(self.y))
            x1, y1 = min(w, int(self.x + self.size)), min(h, int(self.y + self.size))
            if x1 > x0 and y1 > y0:
                cv2.rectangle(img_array, (x0, y0), (x1, y1), 
                              fill_color_with_alpha, -1)
        

        else:
            x0, y0 = int(max(0, min(w, self.x))), int(max(0, min(h, self.y)))
            x1, y1 = int(max(0, min(w, self.x + self.size))), int(max(0, min(h, self.y)))
            x2, y2 = int(max(0, min(w, self.x + self.size/2))), int(max(0, min(h, self.y + self.size)))
            
    
            if not (x0 == x1 and y0 == y1) and not (x0 == x2 and y0 == y2) and not (x1 == x2 and y1 == y2):
                triangle_points = np.array([(x0, y0), (x1, y1), (x2, y2)], np.int32)
                cv2.fillPoly(img_array, [triangle_points], fill_color_with_alpha)
        
        return img_array

class EvolutionaryImageReconstructor:
    def __init__(self, reference_image_path, population_size=200, max_generations=1000):
        """Initialize the evolutionary image reconstruction algorithm."""

        self.reference_image = cv2.imread(reference_image_path)
        if self.reference_image is None:
            raise ValueError(f"Could not read image: {reference_image_path}")
        
        self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
        self.height, self.width, _ = self.reference_image.shape
        self.population_size = population_size
        self.max_generations = max_generations
        

        self.initial_mse = np.mean((self.reference_image - np.zeros_like(self.reference_image)) ** 2)

    def calculate_improvement(self, base_image, new_shape):
        """Calculate the improvement in MSE after adding a new shape."""

        temp_image = base_image.copy()
        temp_image = new_shape.draw(temp_image)
        

        new_mse = np.mean((self.reference_image - temp_image) ** 2)
        return self.initial_mse - new_mse

    def evolve(self, verbose=True):
        """Main evolutionary algorithm to reconstruct the image."""

        current_image = np.zeros_like(self.reference_image)
        
        improvement_history = []
        best_shape = None
        
        for gen in range(self.max_generations):
    
            population = [AdaptiveGeometricShape(self.width, self.height, self.reference_image) for _ in range(self.population_size)]
            
    
            if gen > 0 and best_shape:
                for _ in range(self.population_size // 3):
                    variant = AdaptiveGeometricShape(self.width, self.height, self.reference_image)
                    variant.x = best_shape.x + random.gauss(0, self.width * 0.05)
                    variant.y = best_shape.y + random.gauss(0, self.height * 0.05)
                    variant.size = best_shape.size + random.gauss(0, 5)
                    variant.color = tuple(np.clip(c + random.randint(-30, 30), 0, 255) for c in best_shape.color)
                    variant.alpha = min(1.0, max(0.1, best_shape.alpha + random.gauss(0, 0.1)))
                    variant.shape_type = best_shape.shape_type
                    population.append(variant)
            
    
            improvements = [self.calculate_improvement(current_image, shape) for shape in population]
            
    
            best_idx = np.argmax(improvements)
            best_shape = population[best_idx]
            best_improvement = improvements[best_idx]
            
            improvement_history.append(best_improvement)
            
    
            best_shape.draw(current_image)
            
    
            if gen % 50 == 0 and verbose:
                print(f"Generation {gen} | Improvement: {best_improvement:.4f}")
            
    
            if gen > 100 and sum(improvement_history[-50:]) < 0.1:
                for _ in range(5):
                    random_shape = AdaptiveGeometricShape(self.width, self.height, self.reference_image)
                    random_shape.size = random.randint(2, 10)
                    random_shape.draw(current_image)
                improvement_history = []
        

        cv2.imwrite('reconstructed_image.png', cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
        
        return current_image, improvement_history

def main():
    reconstructor = EvolutionaryImageReconstructor('Mona_Lisa-1.jpg', population_size=200, max_generations=1000)
    reconstructed_image, improvement_history = reconstructor.evolve()

if __name__ == "__main__":
    main()