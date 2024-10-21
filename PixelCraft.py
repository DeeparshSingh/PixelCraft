import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import io, color
from skimage.transform import resize
from skimage import __version__ as skimage_version


# Directory to save the reconstructed images
output_directory = 'd:\\reconstructed_imagesC' #choose your own output path
os.makedirs(output_directory, exist_ok=True)

# Initialize lists to store data for visualization
fitness_progression = []
population_diversity = []

# Load and prepare the target image
target_image_path = 'd:\\chess1.jpg' #enter your own image path
target_image_full = io.imread(target_image_path)
target_image_resized = resize(target_image_full, (50, 50), anti_aliasing=True)

# Ensure the target image is in RGB format
if len(target_image_resized.shape) == 2:
    target_image = color.gray2rgb(target_image_resized)
else:
    target_image = target_image_resized

# Parameters
img_height, img_width = 50, 50
population_size = 50
max_generations = 10000
mutation_rate = 0.06  # Initial mutation rate
elitism_size = 3  # Number of elite individuals preserved

# Initialize population with random images
population = np.random.rand(population_size, img_height, img_width, 3)

# Calculates fitness of the image using SSIM
def calculate_fitness(image):
    if skimage_version >= '0.19':
        return ssim(image, target_image, data_range=image.max() - image.min(), channel_axis=-1)
    else:
        return ssim(image, target_image, data_range=image.max() - image.min(), multichannel=True)

# Selects 2 parents from the population with highest fitness scores
def select_parents(population, fitnesses, num_parents=2, return_indices=False):
    parents_idx = np.argsort(fitnesses)[-num_parents:]
    if return_indices:
        return population[parents_idx], parents_idx
    return population[parents_idx]

# Performs crossover between two parent images to create an offspring
def crossover(parents):
    return (parents[0] + parents[1]) / 2

#  Mutates an image based on a mutation rate
def mutate(image, mutation_rate, parent_fitness=None):
    if parent_fitness is not None:
        mutation_adjustment = (1 - parent_fitness)
        mutation_rate *= mutation_adjustment
    num_mutations = int(img_height * img_width * mutation_rate)
    for _ in range(num_mutations):
        i, j = np.random.randint(img_height), np.random.randint(img_width)
        image[i, j, :] = np.random.rand(3)
    return image

# Generates the next generation of the population
def evolve(population, mutation_rate):
    fitnesses = np.array([calculate_fitness(img) for img in population])
    elite_indices = np.argsort(fitnesses)[-elitism_size:]
    elites = population[elite_indices] # Preserves a certain number of elite individuals from the current population
    next_generation = np.array(elites, copy=True)
    
    while len(next_generation) < population_size:
        parents, parents_idx = select_parents(population, fitnesses, return_indices=True)
        offspring = crossover(parents)
        parent_fitness = np.mean(fitnesses[parents_idx])
        offspring = mutate(offspring, mutation_rate, parent_fitness=parent_fitness)
        next_generation = np.append(next_generation, [offspring], axis=0)
    
    return next_generation, mutation_rate

# Adjusts the mutation rate over generations according to a cooling schedule
def cooling_schedule(mutation_rate, generation, max_generations):
    base_rate = max(0.01, mutation_rate * np.exp(-generation / max_generations)) # Uses an exponential decay function to decrease the mutation rate gradually
    if generation % 20 == 0:
        return min(0.06, base_rate * 4)
    return base_rate


# Evolution loop
for generation in range(max_generations):
    mutation_rate = cooling_schedule(mutation_rate, generation, max_generations)
    population, mutation_rate = evolve(population, mutation_rate)
    best_fitness = np.max([calculate_fitness(img) for img in population])
    print(f"Generation {generation + 1}, Best Fitness: {best_fitness}, Mutation Rate: {mutation_rate}")
    
    # Store data for visualization
    fitness_progression.append(best_fitness)

    # Calculate population diversity 
    population_diversity.append(np.std(population))
    
    # Save reconstructed images after every 250 generations and the final best image
    if (generation + 1) % 250 == 0 or generation == max_generations - 1:
        final_image = population[np.argmax([calculate_fitness(img) for img in population])]
        plt.imshow(final_image)
        plt.title(f"Reconstructed Image - Generation {generation + 1}")
        plt.savefig(os.path.join(output_directory, f'reconstructed_image_generation_{generation + 1}.png'))
        plt.close()

# Plot fitness progression over generations
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_generations + 1), fitness_progression, color='blue', linestyle='-')
plt.title('Fitness Progression Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.savefig(os.path.join(output_directory, 'fitness_progression.png'))
plt.show()

# Plot population diversity over generations
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_generations + 1), population_diversity, color='green', linestyle='--')
plt.title('Population Diversity Over Generations')
plt.xlabel('Generation')
plt.ylabel('Population Diversity')
plt.grid(True)
plt.savefig(os.path.join(output_directory, 'population_diversity.png'))
plt.show()

# Save the final best image
final_image = population[np.argmax([calculate_fitness(img) for img in population])]
plt.imshow(final_image)
plt.title("Final Best Reconstructed Image")
plt.savefig(os.path.join(output_directory, 'final_best_reconstructed_image.png'))
plt.show()