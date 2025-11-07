import random

# -----------------------------
# 1️⃣ Définition du problème
# -----------------------------

# Liste des tâches avec leurs durées
taches = {
    "T1": 5,
    "T2": 3,
    "T3": 2,
    "T4": 4
}

# Contraintes de précédence (clé : tâche, valeur : doit précéder)
precedences = {
    "T3": ["T1"],   # T1 doit précéder T3
    "T4": ["T2"]    # T2 doit précéder T4
}

# -----------------------------
# 2️⃣ Fonctions utilitaires
# -----------------------------

def est_solution_valide(solution):
    """Vérifie si une solution respecte les dépendances."""
    for tache, prereq in precedences.items():
        for p in prereq:
            if solution.index(p) > solution.index(tache):
                return False
    return True


def cout(solution):
    """Calcule le coût (makespan = temps total)."""
    total = 0
    for t in solution:
        total += taches[t]
    return total


def generer_voisins(solution):
    """Génère des solutions voisines en échangeant deux tâches."""
    voisins = []
    n = len(solution)
    for i in range(n):
        for j in range(i+1, n):
            voisin = solution.copy()
            voisin[i], voisin[j] = voisin[j], voisin[i]
            if est_solution_valide(voisin):
                voisins.append(voisin)
    return voisins


# -----------------------------
# 3️⃣ Recherche en Tabou
# -----------------------------

def recherche_tabou(taches, iterations=30, taille_tabou=5):
    # Solution initiale aléatoire valide
    solution_courante = list(taches.keys())
    random.shuffle(solution_courante)
    while not est_solution_valide(solution_courante):
        random.shuffle(solution_courante)

    meilleure_solution = solution_courante.copy()
    meilleur_cout = cout(meilleure_solution)
    liste_tabou = []

    print(f"Solution initiale : {solution_courante} | Coût = {meilleur_cout}\n")

    for it in range(iterations):
        voisins = generer_voisins(solution_courante)

        # Si pas de voisin valide, on s'arrête
        if not voisins:
            break

        meilleur_voisin = None
        meilleur_voisin_cout = float("inf")

        for v in voisins:
            c = cout(v)
            if (v not in liste_tabou) and (c < meilleur_voisin_cout):
                meilleur_voisin = v
                meilleur_voisin_cout = c

        # Mise à jour de la solution courante
        if meilleur_voisin is not None:
            solution_courante = meilleur_voisin
            cout_courant = meilleur_voisin_cout

            # Mise à jour du meilleur global
            if cout_courant < meilleur_cout:
                meilleure_solution = solution_courante.copy()
                meilleur_cout = cout_courant

            # Mise à jour de la liste tabou
            liste_tabou.append(solution_courante)
            if len(liste_tabou) > taille_tabou:
                liste_tabou.pop(0)

        print(f"Iteration {it+1:02d} : {solution_courante} | Coût = {cout_courant}")

    print("\n✅ Meilleure solution trouvée :", meilleure_solution)
    print("💰 Coût minimal :", meilleur_cout)




recherche_tabou(taches)
