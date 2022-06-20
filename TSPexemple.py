# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# Résolution du problème du voyageur de commerce ou TCS à l'aide du recuit simulé
# import de la librairie
import json
import random
import numpy as np
import matplotlib.pyplot as plt

# Données du problème (générées aléatoirement)
NOMBRE_DE_VILLES = 10
NB_GENERATION = 100
MAX_DISTANCE = 2000


def generate_distances():
    distances = np.zeros((NOMBRE_DE_VILLES, NOMBRE_DE_VILLES))
    for ville in range(NOMBRE_DE_VILLES):
        villes = [i for i in range(NOMBRE_DE_VILLES) if not i == ville]
        for vers_la_ville in villes:
            distances[ville][vers_la_ville] = random.randint(50, MAX_DISTANCE)
            distances[vers_la_ville][ville] = distances[ville][vers_la_ville]
    return distances


def export_data(distances):
    with open("data.json", "w") as f:
        json.dump(distances.tolist(), f)


def import_data():
    with open("data.json", "r") as f:
        distances = json.load(f)
    return np.array(distances)


distances = generate_distances()
export_data(distances)
distances = import_data()
print('voici la matrice des distances entres les villes \n', distances)


def cal_distance(solution, distances, NOMBRE_DE_VILLES):
    eval_distance = 0
    for i in range(len(solution)):
        origine, destination = solution[i], solution[(i + 1) % NOMBRE_DE_VILLES]
        eval_distance += distances[origine][destination]
    return eval_distance


def voisinage(solution, NOMBRE_DE_VILLES):
    echange = random.sample(range(NOMBRE_DE_VILLES), 2)
    sol_voisine = solution
    (sol_voisine[echange[0]], sol_voisine[echange[1]]) = (sol_voisine[echange[1]], sol_voisine[echange[0]])
    return sol_voisine


# recuit simulé
solution = random.sample(range(NOMBRE_DE_VILLES), NOMBRE_DE_VILLES)
cout0 = cal_distance(solution, distances, NOMBRE_DE_VILLES)
T = 3500
facteur = 0.90
T_intiale = MAX_DISTANCE / 2
min_sol = solution
cout_min_sol = cout0
historique_cout = []
for i in range(NB_GENERATION):
    historique_cout.append(cout0)
    print('la ', i, 'ème solution = ', solution, ' distance totale= ', cout0, ' température actuelle =', T)
    T = T * facteur
    for j in range(50):
        nouv_sol = voisinage(solution * 1, NOMBRE_DE_VILLES)
        cout1 = cal_distance(nouv_sol, distances, NOMBRE_DE_VILLES)
        if cout1 < cout0:
            cout0 = cout1
            solution = nouv_sol
            if cout1 < cout_min_sol:
                cout_min_sol = cout1
                min_sol = solution
        else:
            x = np.random.uniform()
            if x < np.exp((cout0 - cout1) / T):
                cout0 = cout1
                solution = nouv_sol

print('voici la solution retenue ', min_sol, ' et son coût ', cal_distance(min_sol, distances, NOMBRE_DE_VILLES))

plt.plot(list(range(NB_GENERATION)), historique_cout, label='Valeur maximale')
plt.legend()
plt.title('Evolution du coût à travers les générations')
plt.xlabel('Nbr de générations')
plt.ylabel('Coût')
plt.show()
