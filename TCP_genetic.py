import random
import numpy as np
import matplotlib.pyplot as plt


# 1. PROBLÈME TSP - Définition des villes et distances
class TSP:
    def __init__(self, cities, distances):
        self.cities = cities
        self.distances = distances

    def calculate_distance(self, path):
        """Calcule la distance totale d'un parcours"""
        total_distance = 0
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % len(path)]  # Retour au départ pour le dernier
            total_distance += self.distances[from_city][to_city]
        return total_distance


# 2. ALGORITHME GÉNÉTIQUE
class GeneticAlgorithm:
    def __init__(self, tsp, population_size=50, mutation_rate=0.01, generations=100):
        self.tsp = tsp
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def create_individual(self):
        """Crée un individu aléatoire (permutation des villes)"""
        individual = list(range(len(self.tsp.cities)))
        random.shuffle(individual)
        return individual

    def create_population(self):
        """Crée la population initiale"""
        return [self.create_individual() for _ in range(self.population_size)]

    def calculate_fitness(self, individual):
        """Calcule la fitness d'un individu (1/distance)"""
        distance = self.tsp.calculate_distance(individual)
        return 1.0 / distance  # Plus la distance est petite, plus la fitness est grande

    def selection_roulette(self, population, fitnesses):
        """Sélection par méthode de la roulette"""
        # Calcul des probabilités de sélection
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]

        # Calcul des probabilités cumulées
        cumulative_probs = []
        cumulative_sum = 0
        for prob in probabilities:
            cumulative_sum += prob
            cumulative_probs.append(cumulative_sum)

        # Sélection de deux parents
        parents = []
        for _ in range(2):
            r = random.random()
            for i, cum_prob in enumerate(cumulative_probs):
                if r <= cum_prob:
                    parents.append(population[i])
                    break
        return parents

    def crossover_masque(self, parent1, parent2):
        """Croisement par masque binaire"""
        size = len(parent1)

        # Génération d'un masque aléatoire
        mask = [random.randint(0, 1) for _ in range(size)]

        # Création de l'enfant 1
        child1 = [None] * size

        # Étape 1: Remplir avec les gènes du parent 1 où le masque = 1
        for i in range(size):
            if mask[i] == 1:
                child1[i] = parent1[i]

        # Étape 2: Remplir les trous avec les gènes du parent 2 (dans l'ordre)
        parent2_index = 0
        for i in range(size):
            if child1[i] is None:
                # Trouver la prochaine ville du parent 2 qui n'est pas déjà dans l'enfant
                while parent2[parent2_index] in child1:
                    parent2_index += 1
                child1[i] = parent2[parent2_index]
                parent2_index += 1

        return child1

    def mutate(self, individual):
        """Mutation: échange deux villes aléatoires"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def run(self):
        """Exécute l'algorithme génétique"""
        # Initialisation
        population = self.create_population()
        best_individual = None
        best_fitness = 0
        fitness_history = []

        for generation in range(self.generations):
            # Calcul des fitness
            fitnesses = [self.calculate_fitness(ind) for ind in population]

            # Mise à jour de la meilleure solution
            current_best_fitness = max(fitnesses)
            current_best_index = fitnesses.index(current_best_fitness)

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_index].copy()

            fitness_history.append(1 / best_fitness)  # Stocke la distance

            # Affichage de la progression
            if generation % 20 == 0:
                print(f"Génération {generation}: Meilleure distance = {1 / best_fitness:.2f}")

            # Création de la nouvelle génération
            new_population = []

            # Élitisme: on garde le meilleur individu
            new_population.append(best_individual)

            # Remplissage du reste de la population
            while len(new_population) < self.population_size:
                # Sélection
                parents = self.selection_roulette(population, fitnesses)

                # Croisement
                child = self.crossover_masque(parents[0], parents[1])

                # Mutation
                child = self.mutate(child)

                new_population.append(child)

            population = new_population

        return best_individual, fitness_history


# 3. EXEMPLE D'UTILISATION
def main():
    # Définition d'un problème TSP simple avec 10 villes
    num_cities = 10

    # Création de coordonnées aléatoires pour les villes
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

    # Calcul de la matrice des distances (euclidiennes)
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dx = cities[i][0] - cities[j][0]
                dy = cities[i][1] - cities[j][1]
                distances[i][j] = np.sqrt(dx ** 2 + dy ** 2)

    # Création du problème TSP
    tsp_problem = TSP(list(range(num_cities)), distances)

    # Création et exécution de l'algorithme génétique
    ga = GeneticAlgorithm(
        tsp_problem,
        population_size=50,
        mutation_rate=0.02,
        generations=200
    )

    print("Début de l'algorithme génétique...")
    best_solution, history = ga.run()

    # Résultats
    print(f"\n=== RÉSULTATS ===")
    print(f"Meilleur parcours trouvé: {best_solution}")
    print(f"Distance: {tsp_problem.calculate_distance(best_solution):.2f}")

    # Visualisation
    plt.figure(figsize=(12, 5))

    # Graphique 1: Évolution de la distance
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.title('Évolution de la distance optimale')
    plt.xlabel('Génération')
    plt.ylabel('Distance')
    plt.grid(True)

    # Graphique 2: Visualisation du parcours
    plt.subplot(1, 2, 2)

    # Coordonnées des villes dans l'ordre du meilleur parcours
    best_path_coords = [cities[i] for i in best_solution]
    best_path_coords.append(best_path_coords[0])  # Retour au point de départ

    x_coords = [coord[0] for coord in best_path_coords]
    y_coords = [coord[1] for coord in best_path_coords]

    plt.plot(x_coords, y_coords, 'b-', marker='o', markersize=8)
    plt.title('Meilleur parcours trouvé')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    # Numérotation des villes
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, f' {i}', fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()