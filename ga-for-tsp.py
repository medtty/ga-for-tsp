#Import Statements
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

#Data for Cities
cities_names = [
    "Casablanca", "Marrakech", "Rabat", "Fes", "Tangier", "Agadir", "Oujda", "Meknes", 
    "Kenitra", "Taza", "Essaouira", "Guelmim", "Nador", "El Jadida", "Safi", "Khenifra", 
    "Tiznit", "Beni Mellal", "Taroudant", "Ouarzazate"
]

x = [-7.61138, -7.99205, -6.84165, -5.00229, -5.80214, -9.59811, -1.91282, -5.53763, 
     -6.57001, -4.00767, -9.75435, -10.06667, -2.93330, -8.53700, -9.23000, -6.66667, 
     -9.73320, -6.34300, -8.96700, -6.89500]

y = [33.58831, 31.62867, 34.02088, 34.03724, 35.76785, 30.42776, 34.68940, 33.89379, 
     34.26167, 34.26578, 31.50633, 29.10000, 35.17450, 33.25400, 32.30000, 32.93333, 
     29.69690, 32.36000, 30.46667, 30.91389]

city_coords = dict(zip(cities_names, zip(x, y)))


# Genetic Algorithm parameters
N_POPULATION = 250
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
N_GENERATIONS = 200

#Euclidean Distance Calculation
def euclidean_distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))


#Fitness Function
def fitness(path, city_coords):
    total_distance = sum(euclidean_distance(city_coords[path[i]], city_coords[path[i+1]]) for i in range(len(path)-1))
    total_distance += euclidean_distance(city_coords[path[-1]], city_coords[path[0]])  # Return to start city
    return total_distance


#Crossover Function
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end] = parent1[start:end]
    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = city
    return child


#Mutation Function
def mutate(path, mutation_rate=0.01):
    for i in range(len(path)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(path) - 1)
            path[i], path[j] = path[j], path[i]
    return path


#Main Genetic Algorithm
def genetic_algorithm(city_coords, n_population, crossover_rate, mutation_rate, n_generations):
    population = [random.sample(list(city_coords.keys()), len(city_coords)) for _ in range(n_population)]
    
    for generation in range(n_generations):
        population = sorted(population, key=lambda path: fitness(path, city_coords))
        next_generation = population[:n_population // 10]  # Elitism
        
        for _ in range(int(n_population * crossover_rate)):
            parent1, parent2 = random.choices(population[:50], k=2)
            child = crossover(parent1, parent2)
            next_generation.append(child)
        
        for _ in range(int(n_population * mutation_rate)):
            mutated = mutate(random.choice(next_generation))
            next_generation.append(mutated)
        
        population = next_generation[:n_population]
        
        if generation % 10 == 0:
            best_path = population[0]
            print(f"Generation {generation} - Best fitness: {fitness(best_path, city_coords):.2f}")
    
    return min(population, key=lambda path: fitness(path, city_coords))


#Visualization Function
def plot_path(city_coords, best_path):
    path_coords = [city_coords[city] for city in best_path] + [city_coords[best_path[0]]]
    x_coords, y_coords = zip(*path_coords)
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    plt.plot(x_coords, y_coords, "bo-", markersize=8, label="Path", alpha=0.7)
    
    for city, (x, y) in city_coords.items():
        plt.text(x, y, city, fontsize=12, ha="right", va="bottom")
    
    plt.title("Best Path Found by Genetic Algorithm", fontsize=16)
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


#Running the Genetic Algorithm and Visualizing the Result
best_path = genetic_algorithm(city_coords, N_POPULATION, CROSSOVER_RATE, MUTATION_RATE, N_GENERATIONS)
print(f"Best path found: {best_path}")
plot_path(city_coords, best_path)







"""
----------------------------------------------------------------------------------------------------
Pastel Palette for the visual graph with the introduction
"""
colors = sns.color_palette("pastel", len(cities_names))

city_icons = {
    "Casablanca": "♕",      # Queen
    "Marrakech": "♖",       # Rook
    "Rabat": "♗",           # Bishop
    "Fes": "♘",              # Knight
    "Tangier": "♙",         # Pawn
    "Agadir": "♔",          # King
    "Oujda": "♚",           # Black King
    "Meknes": "♛",          # Black Queen
    "Kenitra": "♜",         # Black Rook
    "Taza": "♝",            # Black Bishop
    "Essaouira": "♞",       # Black Knight
    "Guelmim": "♟",         # Black Pawn
    "Nador": "♕",           # Queen
    "El Jadida": "♖",       # Rook
    "Safi": "♗",            # Bishop
    "Khenifra": "♘",        # Knight
    "Tiznit": "♙",          # Pawn
    "Beni Mellal": "♔",     # King
    "Taroudant": "♚",       # Black King
    "Ouarzazate": "♛"       # Black Queen
}


fig, ax = plt.subplots()

ax.grid(False)  # Disable grid

# Plot each city with its corresponding color and icon
for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    icon = city_icons[city]
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                textcoords='offset points')

    # Connect cities with lines
    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

# Set figure size and display the plot
fig.set_size_inches(16, 12)
plt.title('Random Cities in Morocco', fontsize=20)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.show()

"""
Pastel Palette for the visual graph with the introduction
----------------------------------------------------------------------------------------------------
"""