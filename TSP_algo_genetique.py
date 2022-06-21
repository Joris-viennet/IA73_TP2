# Imports
import json

import numpy as np
import random
import matplotlib.pyplot as plt

from datetime import datetime

# Parameters
NOMBRE_DE_VILLES = 100
MAX_DISTANCE = 2000
n_population = 100
mutation_rate = 0.9
NB_GENERATION = 1000


# Function to compute the distance between two points
def compute_city_distance_coordinates(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def compute_city_distance_names(city_a, city_b, cities_dict):
    return compute_city_distance_coordinates(cities_dict[city_a], cities_dict[city_b])


def generate_distances():
    distances = np.zeros((NOMBRE_DE_VILLES, NOMBRE_DE_VILLES))
    for ville in range(NOMBRE_DE_VILLES):
        villes = [i for i in range(NOMBRE_DE_VILLES) if not i == ville]
        for vers_la_ville in villes:
            distances[ville][vers_la_ville] = random.randint(50, MAX_DISTANCE)
            distances[vers_la_ville][ville] = distances[ville][vers_la_ville]
    return distances


def import_data():
    with open("data.json", "r") as f:
        distances = json.load(f)
    return np.array(distances)


def genesis(city_list, n_population):
    population_set = []
    for i in range(n_population):
        # Randomly generating a new solution
        sol_i = city_list[np.random.choice(list(range(NOMBRE_DE_VILLES)), NOMBRE_DE_VILLES, replace=False)]
        population_set.append(sol_i)
    return np.array(population_set)


def cal_distance(solution, distances, NOMBRE_DE_VILLES):
    eval_distance = 0
    for i in range(len(solution)):
        origine, destination = solution[i], solution[(i + 1) % NOMBRE_DE_VILLES]
        eval_distance += distances[origine][destination]
    return eval_distance


def get_all_fitness(population_set, distance_list):
    fitnes_list = np.zeros(n_population)
    # Looping over all solutions computing the fitness for each solution
    for i in range(n_population):
        fitnes_list[i] = cal_distance(population_set[i], distance_list, NOMBRE_DE_VILLES)
    return fitnes_list


def progenitor_selection(population_set, fitnes_list):
    total_fit = fitnes_list.sum()
    prob_list = fitnes_list / total_fit

    # Notice there is the chance that a progenitor. mates with oneself
    progenitor_list_a = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list,
                                         replace=True)
    progenitor_list_b = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list,
                                         replace=True)
    progenitor_list_a = population_set[progenitor_list_a]
    progenitor_list_b = population_set[progenitor_list_b]
    return np.array([progenitor_list_a, progenitor_list_b])


def mate_progenitors(prog_a, prog_b):
    offspring = prog_a[0:5]
    for city in prog_b:
        if not city in offspring:
            offspring = np.concatenate((offspring, [city]))
    return offspring


def mate_population(progenitor_list):
    new_population_set = []
    for i in range(progenitor_list.shape[1]):
        prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
        offspring = mate_progenitors(prog_a, prog_b)
        new_population_set.append(offspring)
    return new_population_set


def mutate_offspring(offspring):
    for q in range(int(NOMBRE_DE_VILLES * mutation_rate)):
        a = np.random.randint(0, NOMBRE_DE_VILLES)
        b = np.random.randint(0, NOMBRE_DE_VILLES)

        offspring[a], offspring[b] = offspring[b], offspring[a]

    return offspring


def mutate_population(new_population_set):
    mutated_pop = []
    for offspring in new_population_set:
        mutated_pop.append(mutate_offspring(offspring))
    return mutated_pop


def main():
    distances_list = import_data()

    print('voici la matrice des distances entres les villes \n', distances_list)

    cities_names = np.array([i for i in range(NOMBRE_DE_VILLES)])
    print('voici la liste des villes \n', cities_names)

    population_set = genesis(cities_names, n_population)
    print('voici la population initiale \n', population_set)

    fitness_list = get_all_fitness(population_set, distances_list)
    print('voici la liste des fitness \n', fitness_list)

    progenitor_list = progenitor_selection(population_set, fitness_list)
    # print('voici la liste des progeniteurs \n', progenitor_list[0])

    new_population_set = mate_population(progenitor_list)
    print('voici la liste des enfants \n', new_population_set)

    mutated_pop = mutate_population(new_population_set)
    print('voici la liste des enfants mutés \n', mutated_pop)

    best_solution = [-1, np.inf, np.array([])]
    historique_cout = []
    for i in range(NB_GENERATION):
        # if i % 100 == 0: print(i, fitnes_list.min(), fitnes_list.mean(), datetime.now().strftime("%d/%m/%y %H:%M"))
        fitnes_list = get_all_fitness(mutated_pop, distances_list)

        # Saving the best solution
        if fitnes_list.min() < best_solution[1]:
            best_solution[0] = i
            best_solution[1] = fitnes_list.min()
            best_solution[2] = np.array(mutated_pop)[fitnes_list.min() == fitnes_list][0]
        historique_cout.append(best_solution[1])
        progenitor_list = progenitor_selection(population_set, fitnes_list)
        new_population_set = mate_population(progenitor_list)

        mutated_pop = mutate_population(new_population_set)

    print("voici la solution final :", best_solution)

    plt.figure(figsize=(7, 4))
    plt.plot(list(range(NB_GENERATION)), historique_cout, label='Valeur maximale')
    plt.legend()
    plt.title('Evolution du coût à travers les générations')
    plt.xlabel('Nbr de générations')
    plt.ylabel('Coût')
    plt.savefig(f"genetique_{NB_GENERATION}.png")


if __name__ == '__main__':
    main()
